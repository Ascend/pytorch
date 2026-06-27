# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""
Distributed Muon optimizer for Ascend NPU — independent of upstream torch.optim.Muon.

This module provides a standalone _DistributedMuon optimizer class that supports
three Newton-Schulz orthogonalization paths for 2D matrix parameters:

- Standard (non-distributed) NS: for regular tensors, no communication needed.
- Shard(1) distributed NS: for DTensor parameters sharded on dim-1.
  Uses HCCL all-reduce on the decomposed Gram matrix X·X^T = Σ(X_i·X_i^T).
  Zero computational redundancy; each rank holds [m, n/N] of the global [m, n] matrix.
- Shard(0) grouped NS: for DTensor parameters sharded on dim-0.
  Round-robin ownership (param i → rank i % N). All ranks all-gather the gradient; the
  owner rank orthogonalizes (NS) and broadcasts the result, parallelizing NS compute.

Usage:
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard
    from torch_npu.optim import _DistributedMuon

    mesh = init_device_mesh("npu", (world_size,))
    optimizer = _DistributedMuon(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.1,
        process_group=mesh.get_group(),
    )

    # Standard training loop — _DistributedMuon auto-detects DTensor vs regular params
    for input, target in dataset:
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
"""

import math
from collections.abc import MutableMapping
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, Shard
from torch.optim.optimizer import Optimizer, ParamsT


__all__: list[str] = []


# Constants (same as Keller Jordan's Muon)
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Local reimplementations (decoupled from torch.optim._muon)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _to_scalar(lr: float | Tensor) -> float:
    """Convert a possibly-Tensor lr to a Python float."""
    if isinstance(lr, Tensor):
        return lr.item()
    return lr


def _newtonschulz_orthogonalize(
    grad: Tensor,
    ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients are selected to maximize the slope at zero.
    For the purpose of minimizing steps, it turns out to be empirically effective to keep
    increasing the slope at zero even beyond the point where the iteration no longer
    converges all the way to one everywhere on the interval. This iteration therefore
    does not produce UV^T but rather something like US'V^T where S' is diagonal with
    S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model performance at all
    relative to UV^T, where USV^T = G is the SVD.

    Reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    """
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")

    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()

    # Transpose if tall matrix so we compute the smaller Gram [n, n]
    if ortho_grad.size(0) > ortho_grad.size(1):
        ortho_grad = ortho_grad.T

    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))

    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    # Transpose back if we transposed earlier
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T

    return ortho_grad


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Learning rate adjustment for rectangular matrices.

    - "original" (or None): lr * sqrt(max(1, A/B))  — Keller Jordan's scaling
    - "match_rms_adamw": lr * 0.2 * sqrt(max(A, B)) — Moonshot's scaling to match AdamW RMS
    """
    A, B = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0

    return lr * adjusted_ratio


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DTensor inspection helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _is_dtensor_shard(param: Tensor) -> bool:
    """Check if a parameter is a DTensor with any Shard placement."""
    if not isinstance(param, DTensor):
        return False
    for placement in param._spec.placements:
        if isinstance(placement, Shard):
            return True
    return False


def _is_dtensor_shard1(param: Tensor) -> bool:
    """Check if a parameter is a DTensor sharded on dim-1 (Shard(1))."""
    if not isinstance(param, DTensor):
        return False
    for placement in param._spec.placements:
        if isinstance(placement, Shard) and placement.dim == 1:
            return True
    return False


def _get_shard_dim(param: DTensor) -> int:
    """Get the shard dimension from a DTensor placements. Returns 0 by default."""
    for placement in param._spec.placements:
        if isinstance(placement, Shard):
            return placement.dim
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Distributed Newton-Schulz iteration (Shard(1) path)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _distributed_zeropower_via_newtonschulz(
    local_grad: Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    process_group: ProcessGroup,
) -> Tensor:
    """
    Distributed Newton-Schulz iteration for Shard(1) DTensor parameters.

    When the gradient is a local shard of a matrix sharded on dim-1 (each rank holds
    [m, n/N] of a global [m, n] matrix), the Gram matrix X·X^T decomposes as
    Σ(X_i·X_i^T). Each rank computes a partial Gram locally and all-reduces to get
    the full result, enabling distributed computation via HCCL all-reduce instead of
    all-gather.

    No transposition is needed for Shard(1). Unlike the standard NS iteration which
    transposes tall matrices to compute the smaller Gram [n,n], the distributed version
    always computes X·X^T = [m,m] directly via the all-reduce decomposition. This
    produces mathematically identical results because matrix multiplication associativity
    guarantees (X·X^T)·X = X·(X^T·X).

    Args:
        local_grad: Local shard of the gradient tensor, shape [m, n/N].
        ns_coefficients: Coefficients (a, b, c) for the quintic NS polynomial.
        ns_steps: Number of NS iteration steps.
        eps: Small value for numerical stability in norm normalization.
        process_group: ProcessGroup for HCCL all-reduce communication.

    Returns:
        Local shard of the orthogonalized gradient, shape [m, n/N].

    Communication pattern (per optimizer step):
        - 1 HCCL all-reduce for Frobenius norm normalization (scalar)
        - ns_steps HCCL all-reduces for Gram matrix aggregation ([m, m])
    """
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(local_grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")

    a, b, c = ns_coefficients

    ortho_grad = local_grad.bfloat16()

    # Distributed Frobenius norm normalization
    local_norm_sq = ortho_grad.pow(2).sum()
    dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM, group=process_group)
    global_norm = local_norm_sq.sqrt().clamp(min=eps)
    ortho_grad.div_(global_norm)

    # Distributed NS iterations
    for _ in range(ns_steps):
        partial_gram = ortho_grad @ ortho_grad.T
        dist.all_reduce(partial_gram, op=dist.ReduceOp.SUM, group=process_group)

        gram_update = torch.addmm(
            partial_gram, partial_gram, partial_gram, beta=b, alpha=c
        )

        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    return ortho_grad


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _DistributedMuon optimizer class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _DistributedMuon(Optimizer):
    """
    Muon optimizer with distributed Newton-Schulz support for Ascend NPU.

    A standalone optimizer (independent of torch.optim.Muon) that supports three
    orthogonalization paths:

    - Standard NS: for regular (non-DTensor) parameters. No communication needed.
    - Shard(1) distributed NS: for DTensor parameters sharded on dim-1. Uses HCCL
      all-reduce on the decomposed Gram matrix. Zero computational redundancy.
    - Shard(0) grouped NS: for DTensor parameters sharded on dim-0. Round-robin
      ownership (param i → rank i % N). All ranks all-gather; the owner orthogonalizes
      (NS) and broadcasts the result.

    When process_group is None, all parameters use the standard (non-distributed)
    NS path, producing identical results to the upstream Muon optimizer.

    Args:
        params: Iterable of parameters to optimize. Must be 2D matrices.
        lr (float): Learning rate (default: 1e-3).
        weight_decay (float): Weight decay (L2 penalty) (default: 0.1).
        momentum (float): Momentum factor (default: 0.95).
        nesterov (bool): Enables Nesterov momentum (default: True).
        ns_coefficients (tuple): Coefficients (a, b, c) for the quintic NS
            polynomial (default: (3.4445, -4.7750, 2.0315)).
        eps (float): Numerical stability term (default: 1e-7).
        ns_steps (int): Number of Newton-Schulz iteration steps (default: 5).
        adjust_lr_fn (str | None): Learning rate adjustment. One of "original"
            (or None) or "match_rms_adamw" (default: None).
        process_group (ProcessGroup | None): ProcessGroup for distributed NS.
            If None, all parameters use the standard (non-distributed) path.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
        process_group: Optional[ProcessGroup] = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in [
            "original",
            "match_rms_adamw",
        ]:
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "process_group": process_group,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"_DistributedMuon only supports 2D parameters whereas "
                        f"we found a parameter with size: {p.size()}"
                    )

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
    ) -> bool:
        """Prepare params, grads, and momentum buffers for a param group.

        For DTensor sharded parameters, unwraps grad and momentum_buffer
        to local tensors (._local_tensor) for the core optimizer math.
        """
        for p in group["params"]:
            if p.grad is None:
                continue

            if torch.is_complex(p):
                raise RuntimeError("_DistributedMuon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("_DistributedMuon does not support sparse gradients")

            params_with_grad.append(p)

            grad = p.grad
            if isinstance(grad, DTensor) and _is_dtensor_shard(p):
                grad = grad._local_tensor
            grads.append(grad)

            state = self.state[p]

            if "momentum_buffer" not in state:
                if isinstance(p.grad, DTensor) and _is_dtensor_shard(p):
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad._local_tensor, memory_format=torch.preserve_format
                    )
                else:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad, memory_format=torch.preserve_format
                    )

            momentum_buf = state["momentum_buffer"]
            if isinstance(momentum_buf, DTensor) and _is_dtensor_shard(p):
                momentum_buf = momentum_buf._local_tensor
            momentum_bufs.append(momentum_buf)

        return False  # has_complex

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            process_group = group["process_group"]

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_bufs: list[Tensor] = []

            has_complex = self._init_group(
                group, params_with_grad, grads, momentum_bufs,
            )

            _distributed_muon_single_tensor(
                params_with_grad, grads, momentum_bufs,
                lr=lr, weight_decay=weight_decay, momentum=momentum,
                nesterov=group["nesterov"], ns_coefficients=group["ns_coefficients"],
                eps=group["eps"], ns_steps=group["ns_steps"],
                adjust_lr_fn=group["adjust_lr_fn"], has_complex=has_complex,
                process_group=process_group,
            )
        return loss


def _distributed_muon_single_tensor(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
    process_group: Optional[ProcessGroup] = None,
) -> None:
    """Core optimizer math with three-way NS dispatch."""
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = momentum_bufs[i]

        # Momentum update
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        # ── Three-way NS dispatch ──
        if process_group is not None and _is_dtensor_shard(param):
            if _is_dtensor_shard1(param):
                # Shard(1) distributed NS (zero redundancy)
                # Distributed NS via HCCL all-reduce on the decomposed Gram matrix.
                # No all-gather needed; each rank works on its local shard.
                update_local = _distributed_zeropower_via_newtonschulz(
                    update, ns_coefficients, ns_steps, eps, process_group
                )
                update = DTensor.from_local(
                    update_local, param.device_mesh, param.placements,
                )
            else:
                # Shard(0) grouped NS (NS compute parallelized across params)
                # Parameters are assigned to ranks round-robin: param i is
                # handled by rank (i % world_size). All ranks all-gather the
                # full gradient (full_tensor is a collective — every rank must
                # call it); only the owner orthogonalizes (NS) and broadcasts
                # the result. Since different ranks own different parameters,
                # the NS computation is parallelized with zero redundancy.
                rank = dist.get_rank(process_group)
                world_size = dist.get_world_size(process_group)
                owner_rank = i % world_size

                # All ranks participate in the all-gather. (Previously only the
                # owner called full_tensor, which left the owner with just its
                # local shard and desynced the process group.)
                update_dtensor = DTensor.from_local(
                    update, param.device_mesh, param.placements
                )
                update_full = update_dtensor.full_tensor()
                orig_dtype = update_full.dtype

                # NS computes in bf16 internally; keep bf16 through broadcast
                # instead of casting up to fp32 first. This halves the broadcast
                # traffic and is bit-identical numerically because bf16→fp32 is
                # a lossless cast (low 16 mantissa bits are zero either way).
                # The local cast back to orig_dtype happens after the chunk.
                if rank == owner_rank:
                    bf16_full = _newtonschulz_orthogonalize(
                        update_full, ns_coefficients, ns_steps, eps
                    ).contiguous()
                else:
                    bf16_full = torch.empty(
                        update_full.shape, dtype=torch.bfloat16,
                        device=update_full.device,
                    )

                # Owner broadcasts the bf16 orthogonalized result to all ranks.
                dist.broadcast(bf16_full, src=owner_rank, group=process_group)

                shard_dim = _get_shard_dim(param)
                # Cast back to the param dtype only on the local shard (small).
                update_shard = bf16_full.chunk(
                    world_size, dim=shard_dim,
                )[rank].to(orig_dtype).contiguous()
                update = DTensor.from_local(
                    update_shard,
                    param.device_mesh, param.placements,
                )
        else:
            # Standard (non-distributed) NS — identical to upstream Muon
            update = _newtonschulz_orthogonalize(update, ns_coefficients, ns_steps, eps)

        # Adjusted learning rate for rectangular matrices
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        # Decoupled weight decay + update step
        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)
