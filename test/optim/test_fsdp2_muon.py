"""
FSDP2 + _DistributedMuon(2D) + AdamW(non‑2D) — cross-strategy consistency & training.

Tests:
  1. Cross-strategy optimizer-update consistency (standard / Shard(0) / Shard(1)).
  2. FSDP2 end-to-end training for Shard(0) and Shard(1) (via shard_placement_fn).

Self-contained (mp.spawn). Run from a dir WITHOUT a local torch_npu/ source tree:
    python test/optim/test_fsdp2_muon.py                # WORLD_SIZE=2
    WORLD_SIZE=8 python test/optim/test_fsdp2_muon.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.fsdp import fully_shard
from torch_npu.optim import _DistributedMuon

DIM, HIDDEN, BATCH, STEPS = 64, 128, 16, 30
M, N = 64, 128  # wide matrix for consistency (no transpose divergence)
DEFAULT_WORLD_SIZE = 2

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, cond, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if cond:
        PASS_COUNT += 1
        print(f"[PASS] {name}")
    else:
        FAIL_COUNT += 1
        print(f"[FAIL] {name} — {detail}")


def _maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def _local_of(p):
    return p._local_tensor if hasattr(p, "_local_tensor") else p


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM, HIDDEN)
        self.ln = nn.LayerNorm(HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, DIM)

    def forward(self, x):
        return self.fc2(torch.relu(self.ln(self.fc1(x))))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Cross-strategy optimizer-update consistency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _step_strategy(W_full, G_full, mesh, shard_dim):
    """One _DistributedMuon.step() on fixed W, G under the given strategy."""
    pg = mesh.get_group()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if shard_dim is None:
        w = W_full.clone()
        w.grad = G_full.clone()
        opt = _DistributedMuon(
            [w], lr=1e-3, momentum=0.95, weight_decay=0.0, process_group=None,
        )
        opt.step()
        return w.detach().clone()

    local_W = W_full.chunk(world_size, dim=shard_dim)[rank].clone()
    local_G = G_full.chunk(world_size, dim=shard_dim)[rank].clone()
    w = DTensor.from_local(local_W, mesh, [Shard(shard_dim)])
    w.grad = DTensor.from_local(local_G, mesh, [Shard(shard_dim)])
    opt = _DistributedMuon(
        [w], lr=1e-3, momentum=0.95, weight_decay=0.0, process_group=pg,
    )
    opt.step()
    return w.full_tensor().detach().clone()


def test_cross_strategy_consistency(mesh):
    """Standard / Shard(0) / Shard(1) must produce identical updates."""
    print("\n=== Cross-strategy optimizer-update consistency ===")
    torch.manual_seed(42)
    W = torch.randn(M, N, device="npu")
    G = torch.randn(M, N, device="npu")

    W_ref = _step_strategy(W, G, mesh, None)
    W_s0 = _step_strategy(W, G, mesh, 0)
    W_s1 = _step_strategy(W, G, mesh, 1)

    d_s0 = _maxdiff(W_ref, W_s0)
    d_s1 = _maxdiff(W_ref, W_s1)
    d_cross = _maxdiff(W_s0, W_s1)
    print(f"    maxdiff: std-vs-shard0={d_s0:.4f} std-vs-shard1={d_s1:.4f} "
          f"shard0-vs-shard1={d_cross:.4f}")

    for label, diff in [("Shard(0) matches standard", d_s0),
                        ("Shard(1) matches standard", d_s1),
                        ("Shard(0) matches Shard(1)", d_cross)]:
        check(label, diff < 5e-2, f"{diff:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. FSDP2 end-to-end training per shard dim
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _train_one(mesh, shard_dim_2d):
    """FSDP2+Muon+AdamW training; 2D weights sharded on shard_dim_2d."""
    rank = dist.get_rank()
    torch.manual_seed(0)
    model = SmallNet().to(torch.device("npu", rank))

    if shard_dim_2d == 1:
        def placement(p):
            return Shard(1) if p.ndim == 2 else None
        for sub in [model.fc1, model.ln, model.fc2]:
            fully_shard(sub, mesh=mesh, shard_placement_fn=placement)
        fully_shard(model, mesh=mesh, shard_placement_fn=placement)
    else:
        for sub in [model.fc1, model.ln, model.fc2]:
            fully_shard(sub, mesh=mesh)
        fully_shard(model, mesh=mesh)

    muon_params = [p for p in model.parameters() if p.ndim == 2]
    adamw_params = [p for p in model.parameters() if p.ndim != 2]
    muon = _DistributedMuon(
        muon_params, lr=2e-3, momentum=0.95, weight_decay=0.01,
        process_group=mesh.get_group(),
    )
    adamw = torch.optim.AdamW(adamw_params, lr=3e-4, weight_decay=0.01)

    w2d_before = _local_of(muon_params[0]).detach().clone()
    w1d_before = _local_of(adamw_params[0]).detach().clone()

    losses = []
    for step in range(STEPS):
        torch.manual_seed(100 + step)
        x = torch.randn(BATCH, DIM, device="npu")
        target = torch.randn(BATCH, DIM, device="npu")
        out = model(x)
        out_full = out.full_tensor() if hasattr(out, "full_tensor") else out
        loss = ((out_full - target) ** 2).mean()
        loss.backward()
        muon.step()
        adamw.step()
        muon.zero_grad()
        adamw.zero_grad()
        losses.append(loss.item())

    w2d_changed = not torch.allclose(w2d_before, _local_of(muon_params[0]))
    w1d_changed = not torch.allclose(w1d_before, _local_of(adamw_params[0]))
    first, last, best = losses[0], losses[-1], min(losses)

    if rank == 0:
        traj = " ".join(f"{l:.3f}" for l in losses[::4])
        print(f"    trajectory: {traj} ... {last:.3f}")
    print(f"[r{rank}] loss {first:.4f} -> {last:.4f} (best {best:.4f}) "
          f"2D_changed={w2d_changed} non-2D_changed={w1d_changed}", flush=True)
    return best < first, w2d_changed, w1d_changed


def test_fsdp2_end_to_end(mesh):
    """FSDP2 trains for both Shard(0) and Shard(1)."""
    print("\n=== FSDP2 end-to-end per shard dim ===")
    for label, dim in [("Shard(0) (default)", 0), ("Shard(1)", 1)]:
        converges, w2d_ch, w1d_ch = _train_one(mesh, dim)
        check(f"{label} converges (min loss < initial)", converges)
        check(f"{label} 2D params updated", w2d_ch)
        check(f"{label} non-2D params updated", w1d_ch)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Worker + main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _run_worker(rank, world_size):
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    mesh = init_device_mesh("npu", (world_size,))

    try:
        test_cross_strategy_consistency(mesh)
        test_fsdp2_end_to_end(mesh)
        if rank == 0:
            print(f"\nSUMMARY: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    finally:
        dist.destroy_process_group()
    if FAIL_COUNT:
        sys.exit(1)


def main():
    world_size = int(os.environ.get("WORLD_SIZE", DEFAULT_WORLD_SIZE))
    if torch.npu.device_count() < world_size:
        print(f"Skipping: need >= {world_size} NPUs, found {torch.npu.device_count()}.")
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(20000 + os.getpid() % 30000))
    os.environ.setdefault("HCCL_WHITELIST_DISABLE", "1")
    mp.spawn(_run_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
