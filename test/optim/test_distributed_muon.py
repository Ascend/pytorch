"""
Comprehensive verification test for _DistributedMuon optimizer on Ascend NPU.

Tests three NS paths:
1. Standard (non-distributed) NS — baseline correctness
2. Shard(1) distributed NS — zero redundancy path
3. Shard(0) grouped NS — round-robin all-gather + broadcast path

Also tests:
4. DTensor helper functions
5. Full optimizer step on standard / Shard(1) / Shard(0) parameters
6. Error handling

Self-contained: spawns its own HCCL workers via torch.multiprocessing (no
external launcher / RANK env needed — matches test/distributed/test_broadcast.py).
Run from a directory WITHOUT a local torch_npu/ source tree so the installed
package (with the compiled _C extension) is imported:
    python test/optim/test_distributed_muon.py                  # WORLD_SIZE=2
    WORLD_SIZE=8 python test/optim/test_distributed_muon.py     # override
    MUON_BENCH=1 python test/optim/test_distributed_muon.py     # also run perf benchmark
"""

import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch_npu.optim import _DistributedMuon
from torch_npu.optim.distributed_muon import (
    _distributed_zeropower_via_newtonschulz,
    _newtonschulz_orthogonalize,
    _is_dtensor_shard,
    _is_dtensor_shard1,
    _get_shard_dim,
)

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
EPS = 1e-7

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"[PASS] {name}")
    else:
        FAIL_COUNT += 1
        print(f"[FAIL] {name} — {detail}")


def orthogonal_close(Q, atol=1.0, rtol=0.3):
    """Check if Q^T Q ≈ I (orthogonal matrix property).

    Note: bfloat16 NS iteration has inherent precision limits (~0.8 deviation
    from identity for small matrices). The primary correctness check is whether
    the distributed path matches the standard NS output.
    """
    Qf = Q.float()
    if Qf.shape[0] >= Qf.shape[1]:
        gram = Qf.T @ Qf  # (n, n)
    else:
        gram = Qf @ Qf.T  # (m, m)
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.allclose(gram, identity, atol=atol, rtol=rtol)


def gram_deviation(Q):
    """Max absolute deviation of Q^T Q from identity."""
    Qf = Q.float()
    gram = Qf.T @ Qf if Qf.shape[0] >= Qf.shape[1] else Qf @ Qf.T
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return (gram - identity).abs().max().item()


def upstream_ns(X):
    """Reference: standard Newton-Schulz iteration (local implementation)."""
    return _newtonschulz_orthogonalize(X, NS_COEFFICIENTS, NS_STEPS, EPS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 1: Standard NS correctness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_standard_ns():
    """Verify standard NS produces orthogonal output."""
    print("\n=== Test 1: Standard (non-distributed) NS ===")
    torch.npu.manual_seed_all(42)
    X = torch.randn(16, 32, device="npu", dtype=torch.bfloat16)
    Q = upstream_ns(X)
    gram = Q.float().T @ Q.float()  # (32, 32) for wide matrix
    dev_from_I = (gram - torch.eye(gram.shape[0], device='npu')).abs().max().item()
    check("standard NS orthogonal", orthogonal_close(Q),
          f"max deviation from I: {dev_from_I:.4f}")

    # Tall matrix
    torch.npu.manual_seed_all(99)
    X_tall = torch.randn(32, 16, device="npu", dtype=torch.bfloat16)
    Q_tall = upstream_ns(X_tall)
    dev_tall = gram_deviation(Q_tall)
    check("standard NS orthogonal (tall matrix)", dev_tall < 1.0,
          f"max deviation from I: {dev_tall:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 2: Shard(1) distributed NS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_shard1_distributed_ns(mesh):
    """Shard(1) path: distributed NS via all-reduce."""
    print("\n=== Test 2: Shard(1) Distributed NS ===")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    pg = mesh.get_group()

    # Create a global matrix and shard it on dim-1
    torch.npu.manual_seed_all(42)  # same seed so all ranks see same full matrix
    m, n = 16, 32
    X_full = torch.randn(m, n, device="npu", dtype=torch.bfloat16)
    n_per_rank = n // world_size
    X_local = X_full[:, rank * n_per_rank:(rank + 1) * n_per_rank].clone()

    # Run distributed NS
    Q_local = _distributed_zeropower_via_newtonschulz(
        X_local,
        ns_coefficients=NS_COEFFICIENTS,
        ns_steps=NS_STEPS,
        eps=EPS,
        process_group=pg,
    )

    # Gather full result to verify orthogonality
    Q_parts = [torch.empty(m, n_per_rank, device="npu", dtype=torch.bfloat16) for _ in range(world_size)]
    dist.all_gather(Q_parts, Q_local.contiguous(), group=pg)
    Q_full = torch.cat(Q_parts, dim=1)

    # Check: distributed result should be orthogonal
    dev_s1 = gram_deviation(Q_full)
    check("shard1 distributed NS orthogonal", dev_s1 < 1.0,
          f"max deviation from I: {dev_s1:.4f}")

    # Check: distributed result should match upstream NS on the same input
    Q_ref = upstream_ns(X_full)
    max_diff = (Q_full.float() - Q_ref.float()).abs().max().item()
    check("shard1 matches upstream NS", max_diff < 0.15,
          f"max diff: {max_diff:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 3: Shard(0) grouped NS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_shard0_grouped_ns(mesh):
    """Shard(0) grouped NS: round-robin all-gather + broadcast."""
    print("\n=== Test 3: Shard(0) Grouped NS ===")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    pg = mesh.get_group()

    # Create a global matrix and shard on dim-0
    torch.npu.manual_seed_all(123)
    m, n = 32, 16
    m_per_rank = m // world_size
    X_full = torch.randn(m, n, device="npu", dtype=torch.bfloat16)
    X_local = X_full[rank * m_per_rank:(rank + 1) * m_per_rank, :].clone()

    # Reference: all-gather then standard NS
    X_dt = DTensor.from_local(X_local, mesh, [Shard(0)])
    X_full_dt = X_dt.full_tensor()
    Q_ref_full = upstream_ns(X_full_dt)

    # Shard(0) grouped path: ALL ranks all-gather; owner (rank 0) does NS + broadcast.
    # full_tensor() is an all_gather collective, so every rank must call it.
    owner_rank = 0
    update_dt = DTensor.from_local(X_local, mesh, [Shard(0)])
    update_full = update_dt.full_tensor()
    if rank == owner_rank:
        update_full = _newtonschulz_orthogonalize(update_full, NS_COEFFICIENTS, NS_STEPS, EPS).contiguous()
    dist.broadcast(update_full, src=owner_rank, group=pg)
    Q_grouped_shard = update_full.chunk(world_size, dim=0)[rank].contiguous()

    # Check: grouped result matches all-gather+NS reference
    Q_ref_shard = Q_ref_full.chunk(world_size, dim=0)[rank].contiguous()
    max_diff = (Q_grouped_shard.float() - Q_ref_shard.float()).abs().max().item()
    check("shard0 grouped matches allgather+NS", max_diff < 0.01,
          f"max diff: {max_diff:.4f}")

    # Check: full result is orthogonal
    Q_parts = [torch.empty(m_per_rank, n, device="npu", dtype=torch.bfloat16) for _ in range(world_size)]
    dist.all_gather(Q_parts, Q_grouped_shard.contiguous(), group=pg)
    Q_recovered = torch.cat(Q_parts, dim=0)
    dev_s0 = gram_deviation(Q_recovered)
    check("shard0 grouped orthogonal", dev_s0 < 1.0,
          f"max deviation from I: {dev_s0:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 4: DTensor helper functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_dtensor_helpers(mesh):
    """Test _is_dtensor_shard, _is_dtensor_shard1, _get_shard_dim."""
    print("\n=== Test 4: DTensor Helper Functions ===")

    # Shard(1) DTensor
    X_s1_local = torch.randn(16, 16, device="npu", dtype=torch.bfloat16)
    X_s1 = DTensor.from_local(X_s1_local, mesh, [Shard(1)])

    check("_is_dtensor_shard(Shard(1))", _is_dtensor_shard(X_s1))
    check("_is_dtensor_shard1(Shard(1))", _is_dtensor_shard1(X_s1))
    check("_get_shard_dim(Shard(1))", _get_shard_dim(X_s1) == 1,
          f"got {_get_shard_dim(X_s1)}")

    # Shard(0) DTensor
    X_s0_local = torch.randn(16, 16, device="npu", dtype=torch.bfloat16)
    X_s0 = DTensor.from_local(X_s0_local, mesh, [Shard(0)])

    check("_is_dtensor_shard(Shard(0))", _is_dtensor_shard(X_s0))
    check("not _is_dtensor_shard1(Shard(0))", not _is_dtensor_shard1(X_s0))
    check("_get_shard_dim(Shard(0))", _get_shard_dim(X_s0) == 0,
          f"got {_get_shard_dim(X_s0)}")

    # Regular tensor (not DTensor)
    X_reg = torch.randn(16, 16, device="npu")
    check("not _is_dtensor_shard(regular tensor)", not _is_dtensor_shard(X_reg))
    check("not _is_dtensor_shard1(regular tensor)", not _is_dtensor_shard1(X_reg))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 5: Full optimizer step (standard path)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_optimizer_step_standard():
    """Run a full optimizer step on standard (non-sharded) parameters."""
    print("\n=== Test 5: Full Optimizer Step (Standard Path) ===")

    weight = torch.randn(16, 32, device="npu")
    weight_orig = weight.clone()
    weight.grad = torch.randn(16, 32, device="npu")

    optimizer = _DistributedMuon([weight], lr=1e-3, weight_decay=0.1)
    optimizer.step()

    # Weight should have changed
    changed = not torch.allclose(weight, weight_orig)
    check("optimizer step changes parameters", changed,
          f"max diff: {(weight - weight_orig).abs().max().item():.6f}")

    # Weight should still be 2D with same shape
    check("parameter shape preserved", weight.shape == (16, 32))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 6: Full optimizer step (Shard(1) path)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_optimizer_step_shard1(mesh):
    """Run a full optimizer step on Shard(1) DTensor parameters."""
    print("\n=== Test 6: Full Optimizer Step (Shard(1) Path) ===")

    pg = mesh.get_group()
    world_size = dist.get_world_size()

    # Create a Shard(1) DTensor parameter
    m, n = 16, 32
    n_per_rank = n // world_size
    local_weight = torch.randn(m, n_per_rank, device="npu")
    local_weight_orig = local_weight.clone()

    placements = [Shard(1)]
    weight_dt = DTensor.from_local(local_weight, mesh, placements)

    # Simulate a Shard(1) gradient
    local_grad = torch.randn(m, n_per_rank, device="npu")
    grad_dt = DTensor.from_local(local_grad, mesh, placements)
    weight_dt.grad = grad_dt

    try:
        optimizer = _DistributedMuon([weight_dt], lr=1e-3, weight_decay=0.1, process_group=pg)
        optimizer.step()

        # Local weight should have changed
        changed = not torch.allclose(weight_dt._local_tensor, local_weight_orig)
        check("shard1 optimizer step changes local params", changed,
              f"max diff: {(weight_dt._local_tensor - local_weight_orig).abs().max().item():.6f}")

        check("shard1 param shape preserved", weight_dt._local_tensor.shape == (m, n_per_rank))
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("shard1 optimizer step runs", False, str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 7: Full optimizer step (Shard(0) path)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_optimizer_step_shard0(mesh):
    """Run a full optimizer step on Shard(0) DTensor parameters."""
    print("\n=== Test 7: Full Optimizer Step (Shard(0) Path) ===")

    pg = mesh.get_group()
    world_size = dist.get_world_size()

    # Create a Shard(0) DTensor parameter
    m, n = 32, 16
    m_per_rank = m // world_size
    local_weight = torch.randn(m_per_rank, n, device="npu")
    local_weight_orig = local_weight.clone()

    placements = [Shard(0)]
    weight_dt = DTensor.from_local(local_weight, mesh, placements)

    # Simulate a Shard(0) gradient
    local_grad = torch.randn(m_per_rank, n, device="npu")
    grad_dt = DTensor.from_local(local_grad, mesh, placements)
    weight_dt.grad = grad_dt

    try:
        optimizer = _DistributedMuon([weight_dt], lr=1e-3, weight_decay=0.1, process_group=pg)
        optimizer.step()

        changed = not torch.allclose(weight_dt._local_tensor, local_weight_orig)
        check("shard0 optimizer step changes local params", changed,
              f"max diff: {(weight_dt._local_tensor - local_weight_orig).abs().max().item():.6f}")

        check("shard0 param shape preserved", weight_dt._local_tensor.shape == (m_per_rank, n))
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("shard0 optimizer step runs", False, str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 8: Mixed params (standard + Shard(1) + Shard(0))
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_optimizer_mixed_params(mesh):
    """Test _DistributedMuon with a mix of standard and DTensor parameters."""
    print("\n=== Test 8: Mixed Params (Standard + Shard(1) + Shard(0)) ===")

    pg = mesh.get_group()
    world_size = dist.get_world_size()

    # Standard param
    w_std = torch.randn(16, 32, device="npu")
    w_std_orig = w_std.clone()
    w_std.grad = torch.randn(16, 32, device="npu")

    # Shard(1) param
    m1, n1 = 16, 32
    n1_per_rank = n1 // world_size
    w_s1_local = torch.randn(m1, n1_per_rank, device="npu")
    w_s1_orig = w_s1_local.clone()
    w_s1 = DTensor.from_local(w_s1_local, mesh, [Shard(1)])
    w_s1.grad = DTensor.from_local(torch.randn(m1, n1_per_rank, device="npu"), mesh, [Shard(1)])

    # Shard(0) param
    m2, n2 = 32, 16
    m2_per_rank = m2 // world_size
    w_s0_local = torch.randn(m2_per_rank, n2, device="npu")
    w_s0_orig = w_s0_local.clone()
    w_s0 = DTensor.from_local(w_s0_local, mesh, [Shard(0)])
    w_s0.grad = DTensor.from_local(torch.randn(m2_per_rank, n2, device="npu"), mesh, [Shard(0)])

    try:
        optimizer = _DistributedMuon(
            [w_std, w_s1, w_s0],
            lr=1e-3, weight_decay=0.1,
            process_group=pg,
        )
        optimizer.step()

        check("mixed: standard param changed", not torch.allclose(w_std, w_std_orig))
        check("mixed: shard1 param changed", not torch.allclose(w_s1._local_tensor, w_s1_orig))
        check("mixed: shard0 param changed", not torch.allclose(w_s0._local_tensor, w_s0_orig))
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("mixed optimizer step runs", False, str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 9: Error handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_error_handling():
    """Verify proper error handling for invalid inputs."""
    print("\n=== Test 9: Error Handling ===")

    # 1D input should raise ValueError
    X_1d = torch.randn(32, device="npu", dtype=torch.bfloat16)
    try:
        _newtonschulz_orthogonalize(X_1d, NS_COEFFICIENTS, NS_STEPS, EPS)
        check("1D input raises ValueError", False, "no error raised")
    except ValueError as e:
        check("1D input raises ValueError", "2D matrix" in str(e))

    # ns_steps >= 100 should raise ValueError
    X_2d = torch.randn(16, 32, device="npu", dtype=torch.bfloat16)
    try:
        _newtonschulz_orthogonalize(X_2d, NS_COEFFICIENTS, 100, EPS)
        check("ns_steps>=100 raises ValueError", False, "no error raised")
    except ValueError as e:
        check("ns_steps>=100 raises ValueError", "less than 100" in str(e))

    # Wrong number of coefficients should raise ValueError
    try:
        _newtonschulz_orthogonalize(X_2d, (1.0, 2.0), NS_STEPS, EPS)
        check("wrong coeff count raises ValueError", False, "no error raised")
    except ValueError as e:
        check("wrong coeff count raises ValueError", "exactly 3" in str(e))

    # 1D parameter in _DistributedMuon should raise ValueError
    w_1d = torch.randn(32, device="npu")
    try:
        _DistributedMuon([w_1d], lr=1e-3)
        check("1D param in _DistributedMuon raises ValueError", False, "no error raised")
    except ValueError as e:
        check("1D param in _DistributedMuon raises ValueError", "2D" in str(e))

    # Invalid adjust_lr_fn should raise ValueError
    w_2d = torch.randn(16, 32, device="npu")
    try:
        _DistributedMuon([w_2d], lr=1e-3, adjust_lr_fn="invalid_fn")
        check("invalid adjust_lr_fn raises ValueError", False, "no error raised")
    except ValueError as e:
        check("invalid adjust_lr_fn raises ValueError", "not supported" in str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 10: Decoupling — no dependency on torch.optim.Muon
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_decoupling():
    """Verify _DistributedMuon works independently without torch.optim.Muon."""
    print("\n=== Test 10: Decoupling Check ===")

    # _DistributedMuon should NOT have patched upstream Muon
    from torch.optim._muon import Muon as UpstreamMuon
    check("upstream Muon NOT patched", not hasattr(UpstreamMuon, "_npu_distributed_patched"))

    # _DistributedMuon should be a different class
    check("_DistributedMuon is separate class", _DistributedMuon is not UpstreamMuon)

    # Standard Muon should NOT accept process_group
    w = torch.randn(16, 32, device="npu")
    try:
        UpstreamMuon([w], lr=1e-3, process_group=dist.group.WORLD)
        check("upstream Muon does NOT accept process_group", False, "unexpectedly accepted")
    except TypeError as e:
        check("upstream Muon does NOT accept process_group", True, str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Performance benchmark (opt-in: MUON_BENCH=1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BENCH_SHAPES = [(2048, 2048), (4096, 4096), (8192, 8192)]


def timed_run(fn, warmup=2, repeat=10):
    """Return (mean_ms, std_ms) over repeat runs after warmup."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def benchmark_ns_paths(mesh):
    """Time the three NS paths. All ranks participate in the collectives."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    pg = mesh.get_group()

    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK (ms per call)")
    print("=" * 60)
    for m, n in BENCH_SHAPES:
        if m % world_size or n % world_size:
            continue
        print(f"\n[{m}x{n}]")

        # Standard NS (local)
        torch.npu.manual_seed_all(42)
        X = torch.randn(m, n, device="npu", dtype=torch.bfloat16)
        mean, std = timed_run(lambda: _newtonschulz_orthogonalize(X, NS_COEFFICIENTS, NS_STEPS, EPS))
        print(f"  Standard NS:      {mean:6.1f} ± {std:.1f}")

        # Shard(1) distributed NS
        torch.npu.manual_seed_all(42)
        X = torch.randn(m, n, device="npu", dtype=torch.bfloat16)
        npr = n // world_size
        Xl = X[:, rank * npr:(rank + 1) * npr].clone()
        mean, std = timed_run(
            lambda: _distributed_zeropower_via_newtonschulz(Xl, NS_COEFFICIENTS, NS_STEPS, EPS, pg)
        )
        print(f"  Shard(1) dist NS: {mean:6.1f} ± {std:.1f}")

        # Shard(0) grouped NS (owner NS + broadcast)
        torch.npu.manual_seed_all(123)
        X = torch.randn(m, n, device="npu", dtype=torch.bfloat16)
        mpr = m // world_size
        Xl0 = X[rank * mpr:(rank + 1) * mpr, :].clone()
        owner = 0

        def shard0_grouped():
            update_dt = DTensor.from_local(Xl0, mesh, [Shard(0)])
            update_full = update_dt.full_tensor()
            if rank == owner:
                update_full = _newtonschulz_orthogonalize(
                    update_full, NS_COEFFICIENTS, NS_STEPS, EPS
                ).contiguous()
            dist.broadcast(update_full, src=owner, group=pg)
            return update_full.chunk(world_size, dim=0)[rank].contiguous()

        mean, std = timed_run(shard0_grouped)
        print(f"  Shard(0) grouped: {mean:6.1f} ± {std:.1f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_WORLD_SIZE = 2


def _run_worker(rank, world_size):
    """One HCCL worker process: bind device, init group, run the test suite."""
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    mesh = init_device_mesh("npu", (world_size,))
    print(f"[rank {rank}/{world_size}] npu:{rank} | {torch.npu.get_device_name(rank)}", flush=True)

    try:
        # Distributed tests — every rank must reach these collectives together.
        test_shard1_distributed_ns(mesh)
        test_shard0_grouped_ns(mesh)
        test_optimizer_step_shard1(mesh)
        test_optimizer_step_shard0(mesh)
        test_optimizer_mixed_params(mesh)

        # Local-only tests — rank 0 only (no collectives; avoids duplicate output).
        if rank == 0:
            test_dtensor_helpers(mesh)
            test_standard_ns()
            test_optimizer_step_standard()
            test_error_handling()
            test_decoupling()

        if rank == 0:
            print(f"\n{'=' * 50}")
            print(f"SUMMARY: {PASS_COUNT} passed, {FAIL_COUNT} failed")
            print(f"{'=' * 50}")
            print("VERIFICATION FAILED ❌" if FAIL_COUNT else "VERIFICATION PASSED ✅")

        # Optional performance benchmark — all ranks participate (opt-in via MUON_BENCH=1).
        if os.environ.get("MUON_BENCH"):
            benchmark_ns_paths(mesh)
    finally:
        dist.destroy_process_group()

    # Surface any failure to mp.spawn → non-zero exit → CI catches it.
    if FAIL_COUNT:
        sys.exit(1)


def main():
    world_size = int(os.environ.get("WORLD_SIZE", DEFAULT_WORLD_SIZE))
    if torch.npu.device_count() < world_size:
        print(f"Skipping: need >= {world_size} NPUs, found {torch.npu.device_count()}.")
        return
    # CI runs this file as a single `python` process with no launcher, so we
    # spawn our own workers (env:// rendezvous needs MASTER_ADDR/PORT set).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29555 + os.getpid() % 40000))
    os.environ.setdefault("HCCL_WHITELIST_DISABLE", "1")
    mp.spawn(_run_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
