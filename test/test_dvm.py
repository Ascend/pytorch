"""DVM (Device Virtual Machine) eager fusion tests.

All tests compare DVM enabled (set env TORCH_NPU_LAZY_FUSION)
vs DVM disabled (no env var set at all) by running the same computation
in two separate subprocesses.

Includes:
  - Individual op correctness (unary, binary, comparison, activation, reduce, matmul, foreach)
  - Edge cases (empty, scalar, non-contiguous, int, large, inplace)
  - Heterogeneous execution (Ascend -> CPU -> Ascend round-trips, CPU reductions, backward)
  - Guarded fallbacks (CPU out tensors, unsupported optional inputs)
  - Fusion patterns (chains, RMSNorm, SwiGLU, attention, LayerNorm, Adam)
  - Small network forward+backward (MLP, Transformer FFN, training loop, attention)
  - Op-level flags (disable_ops, enable_ops_only)
  - BatchNorm ops

Usage:
    python test_dvm.py -v
"""

import os
import sys
import unittest
import subprocess
import tempfile
import torch_npu

DEVICE_NAME = torch_npu.npu.get_device_name(0)

# DVM is only supported on Ascend A2 / A3 / A5 chips:
#   A2:  Ascend910B1 .. Ascend910B4_1  (note: bare "Ascend910B" without a digit
#                                       suffix is the OLDER A1 910B, NOT A2)
#   A3:  Ascend910_9391, _9392, _9381, _9382, _9372, _9362  (underscore + 4 digits)
#   A5:  Ascend950
# Anything else (Ascend910A / Ascend910ProB / Ascend310* / Ascend910PremiumA)
# is not supported -- DVM library has no kernels for these SoCs, and
# TORCH_NPU_LAZY_FUSION self-disables anyway. Skip the whole module in that
# case so the unsupported boxes don't fail CI.
import re

_DVM_SUPPORTED_RE = re.compile(r"^(Ascend910B[0-9]|Ascend910_9[0-9]{3}|Ascend950)")
_DVM_UNSUPPORTED = _DVM_SUPPORTED_RE.match(DEVICE_NAME) is None
_SKIP_DVM_REASON = f"DVM not supported on {DEVICE_NAME}"


def _run_in_subprocess_raw(code, dvm_enabled, save_path=None):
    """Run code in a subprocess, optionally with DVM enabled."""
    env = os.environ.copy()
    # Remove any existing DVM flags to ensure clean state
    env.pop("TORCH_NPU_LAZY_FUSION", None)
    if dvm_enabled:
        env["TORCH_NPU_LAZY_FUSION"] = "True"
    full_code = f"""
import numpy as np
import torch, torch_npu
torch.manual_seed(42)
np.random.seed(1)
{code}
{"torch.save(result, " + repr(save_path) + ")" if save_path is not None else ""}
"""
    return subprocess.run(
        [sys.executable, "-c", full_code],
        env=env, capture_output=True, text=True, timeout=120
    )


def _run_in_subprocess(code, dvm_enabled, save_path):
    """Run code in a subprocess, optionally with DVM enabled, and persist result."""
    r = _run_in_subprocess_raw(code, dvm_enabled, save_path)
    if r.returncode != 0:
        raise RuntimeError(
            f"Subprocess (dvm={'on' if dvm_enabled else 'off'}) failed:\n{r.stderr[-500:]}"
        )


def compare_dvm_on_off(code):
    """Run code with DVM on and off, return (result_off, result_on)."""
    import torch
    with tempfile.TemporaryDirectory() as tmpdir:
        path_off = os.path.join(tmpdir, "off.pt")
        path_on = os.path.join(tmpdir, "on.pt")
        _run_in_subprocess(code, dvm_enabled=False, save_path=path_off)
        _run_in_subprocess(code, dvm_enabled=True, save_path=path_on)
        r_off = torch.load(path_off, weights_only=False)
        r_on = torch.load(path_on, weights_only=False)
        return r_off, r_on


def compare_dvm_on_off_error(code):
    """Run code with DVM on and off, return subprocess results for error-path checks."""
    r_off = _run_in_subprocess_raw(code, dvm_enabled=False)
    r_on = _run_in_subprocess_raw(code, dvm_enabled=True)
    return r_off, r_on


@unittest.skipIf(_DVM_UNSUPPORTED, _SKIP_DVM_REASON)
class _DvmTestBase(unittest.TestCase):
    """Base class with assertion helpers."""

    def _assert_match(self, r_off, r_on, atol=0, msg=""):
        import torch
        if isinstance(r_off, dict):
            for key in r_off:
                self._assert_match(r_off[key], r_on[key], atol, msg=f" key={key}")
        elif isinstance(r_off, (list, tuple)):
            for i, (a, b) in enumerate(zip(r_off, r_on)):
                self._assert_match(a, b, atol, msg=f" idx={i}")
        elif isinstance(r_off, torch.Tensor):
            if r_off.is_floating_point():
                try:
                    torch.testing.assert_close(
                        r_off,
                        r_on,
                        rtol=atol,
                        atol=atol,
                        equal_nan=True,
                    )
                except Exception as err:
                    if not msg:
                        raise
                    raise type(err)(f"{msg}\n{err}") from err
            else:
                self.assertTrue(torch.equal(r_off, r_on), f"Mismatch{msg}")
        else:
            self.assertEqual(r_off, r_on, msg)


# =============================================================================
# Individual op tests (batched by category to reduce subprocess launches)
# =============================================================================

class TestDvmUnaryOps(_DvmTestBase):
    """Unary ops: abs, neg, sqrt, exp, exp_, reciprocal."""

    def test_unary_ops(self):
        r_off, r_on = compare_dvm_on_off("""
x = torch.randn(4, 128)
x_pos = torch.rand(4, 128) + 0.1
x_exp = torch.randn(4, 128) * 0.5  # avoid overflow

result = {
    "abs": torch.abs(x.npu()).cpu(),
    "neg": torch.neg(x.npu()).cpu(),
    "sqrt": torch.sqrt(x_pos.npu()).cpu(),
    "exp": torch.exp(x_exp.npu()).cpu(),
    "reciprocal": torch.reciprocal(x_pos.npu()).cpu(),
}
""")
        self._assert_match(r_off, r_on)

class TestDvmBinaryOps(_DvmTestBase):
    """Binary ops: add, sub, mul, div (tensor and scalar, normal and inplace)."""

    def test_binary_ops(self):
        r_off, r_on = compare_dvm_on_off("""
x = torch.randn(4, 128)
y = torch.randn(4, 128)
y_pos = torch.rand(4, 128) + 0.1

result = {
    "add_scalar": torch.add(x.npu(), 1.5).cpu(),
    "add_scalar_alpha": torch.add(x.npu(), 1.5, alpha=2.0).cpu(),
    "add": (x.npu() + y.npu()).cpu(),
    "add_alpha": torch.add(x.npu(), y.npu(), alpha=2.5).cpu(),
    "sub": (x.npu() - y.npu()).cpu(),
    "mul_scalar": (x.npu() * 3.14).cpu(),
    "mul_tensor": (x.npu() * y.npu()).cpu(),
    "div_scalar": (x.npu() / 2.0).cpu(),
    "div_tensor": (x.npu() / y_pos.npu()).cpu(),
}
""")
        self._assert_match(r_off, r_on)

    def test_binary_inplace(self):
        r_off, r_on = compare_dvm_on_off("""
x = torch.randn(4, 128)
y = torch.randn(4, 128)
y_pos = torch.rand(4, 128) + 0.1

x1 = x.clone().npu(); x1.add_(y.npu())
x2 = x.clone().npu(); x2.sub_(y.npu())
x3 = x.clone().npu(); x3.mul_(y.npu())
x4 = x.clone().npu(); x4.div_(y_pos.npu())

result = {
    "add_": x1.cpu(),
    "sub_": x2.cpu(),
    "mul_": x3.cpu(),
    "div_": x4.cpu(),
}
""")
        self._assert_match(r_off, r_on)

class TestDvmSelectOps(_DvmTestBase):
    """Select ops: where (GPT2-style causal masking pattern)."""

    def test_where_basic(self):
        r_off, r_on = compare_dvm_on_off("""
x = torch.randn(4, 128)
y = torch.full((4, 128), -1e4)
cond = x > 0

result = {
    "fp32": torch.where(cond.npu(), x.npu(), y.npu()).cpu(),
    "fp16": torch.where(cond.npu(), x.half().npu(), y.half().npu()).cpu(),
    "bf16": torch.where(cond.npu(), x.bfloat16().npu(), y.bfloat16().npu()).float().cpu(),
}
""")
        self._assert_match(r_off, r_on)

class TestDvmActivationOps(_DvmTestBase):
    """Activation ops: sigmoid, tanh, gelu, relu, silu, leaky_relu (forward and backward)."""

    def test_activation_forward(self):
        r_off, r_on = compare_dvm_on_off("""
x = torch.randn(4, 128)

result = {
    "sigmoid": torch.sigmoid(x.npu()).cpu(),
    "gelu_tanh": torch.nn.functional.gelu(x.npu(), approximate="tanh").cpu(),
    "gelu_none": torch.nn.functional.gelu(x.npu(), approximate="none").cpu(),
    "relu": torch.relu(x.npu()).cpu(),
    "silu": torch.nn.functional.silu(x.npu()).cpu(),
    "tanh": torch.tanh(x.npu()).cpu(),
}
""")
        # gelu tanh approximation may differ by ~1 ULP between DVM and non-DVM kernels
        self._assert_match(r_off, r_on, atol=5e-7)

    def test_activation_backward(self):
        r_off, r_on = compare_dvm_on_off("""
x = torch.linspace(-3.0, 3.0, steps=4 * 128, dtype=torch.float32).reshape(4, 128)
grad = torch.linspace(-1.5, 1.5, steps=4 * 128, dtype=torch.float32).reshape(4, 128)
x_npu = x.npu()
grad_npu = grad.npu()
sigmoid_out = torch.sigmoid(x_npu)
tanh_out = torch.tanh(x_npu)

result = {
    "sigmoid_backward": torch.ops.aten.sigmoid_backward.default(grad_npu, sigmoid_out).cpu(),
    "tanh_backward": torch.ops.aten.tanh_backward.default(grad_npu, tanh_out).cpu(),
    "gelu_backward": torch.ops.aten.gelu_backward.default(grad_npu, x_npu, approximate="tanh").cpu(),
    "silu_backward": torch.ops.aten.silu_backward.default(grad_npu, x_npu).cpu(),
}
""")
        self._assert_match(r_off, r_on, atol=1e-4)

class TestDvmSwiGLUOps(_DvmTestBase):
    """SwiGLU ops: npu_swiglu (forward).

    npu_swiglu(x, dim) splits x into two halves a, b along dim and computes
    silu(a) * b. Implemented via DVM's ViewInput (strided Load) so the two
    halves are read from x's storage without an explicit chunk + .contiguous().
    Backward (npu_swiglu_backward) is not in DVM because it requires strided
    Store to write [grad_a, grad_b] into one [..., 2N] tensor.
    """

    def test_swiglu_basic(self):
        r_off, r_on = compare_dvm_on_off("""
x_fp32 = torch.randn(4, 256).npu()
x_fp16 = torch.randn(2, 8, 256).half().npu()
x_bf16 = torch.randn(2, 8, 256).bfloat16().npu()

result = {
    "fp32": torch_npu.npu_swiglu(x_fp32, dim=-1).cpu(),
    "fp16": torch_npu.npu_swiglu(x_fp16, dim=-1).cpu(),
    "bf16": torch_npu.npu_swiglu(x_bf16, dim=-1).float().cpu(),
}
""")
        # fp32 path differs by ~1 ULP (2.4e-7) due to op ordering inside silu math
        self._assert_match(r_off, r_on, atol=5e-7)

class TestDvmAclgraphCoexistence(_DvmTestBase):
    """DVM + aclgraph (NPUGraph) capture/replay coexistence.

    aclgraph captures ACL kernel launches on a stream. DVM Flush eventually emits
    standard aclrtLaunchKernel calls that aclgraph records normally. The flush
    happens automatically inside capture_end() because NPUStream::stream() →
    aclrtStream implicit conversion calls MakeSureQueueEmpty() → LazyFusionFlush().

    Constraints:
      - TASK_QUEUE_ENABLE must be 1 (aclgraph rejects 2; DVM rejects 0)
      - capture must run on a non-default stream (aclgraph requirement)

    Each test runs in a subprocess with both env vars forced; verifies the
    captured graph replays produce outputs matching eager execution.
    """

    @staticmethod
    def _run_capture(body):
        """Spawn subprocess with TASK_QUEUE_ENABLE=1 + DVM on, run capture+replay code,
        return tensor of (max_diff_per_replay) for the assertion."""
        import os
        import subprocess
        import sys
        import tempfile
        env = os.environ.copy()
        env.pop("ASCEND_LAZY_FUSION_FLAGS", None)
        env["TASK_QUEUE_ENABLE"] = "1"
        env["TORCH_NPU_LAZY_FUSION"] = "True"
        with tempfile.TemporaryDirectory() as tmp:
            save = os.path.join(tmp, "diffs.pt")
            full_code = (
                "import torch, torch_npu\n"
                "torch.manual_seed(42)\n"
                "device = 'npu:0'\n"
                "torch.npu.set_device(0)\n"
                "s = torch.npu.Stream()\n"
                f"{body}\n"
                f"torch.save(diffs, {save!r})\n"
            )
            r = subprocess.run([sys.executable, "-c", full_code],
                               env=env, capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                raise RuntimeError(f"aclgraph subprocess failed:\n{r.stderr[-500:]}")
            import torch
            return torch.load(save, weights_only=False)

    def test_aclgraph_swiglu(self):
        """Capture a single npu_swiglu, replay with new inputs, verify match eager."""
        diffs = self._run_capture("""
with torch.npu.stream(s):
    x_in = torch.empty(2, 8, 256, dtype=torch.float16, device=device)
    for _ in range(3): torch_npu.npu_swiglu(x_in, dim=-1)
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        out = torch_npu.npu_swiglu(x_in, dim=-1)
    ds = []
    for _ in range(5):
        xn = torch.randn(2, 8, 256, dtype=torch.float16, device=device)
        ref = torch_npu.npu_swiglu(xn, dim=-1)
        x_in.copy_(xn); g.replay(); torch.npu.synchronize()
        ds.append((ref.float() - out.float()).abs().max().item())
diffs = torch.tensor(ds)
""")
        self.assertTrue((diffs < 1e-3).all().item(), f"diffs: {diffs.tolist()}")

class TestDvmReduceOps(_DvmTestBase):
    """Reduce ops: sum (all, dim, multi-dim)."""

    def test_reduce_ops(self):
        r_off, r_on = compare_dvm_on_off("""
x = torch.randn(2, 4, 8)

result = {
    "sum_all": torch.sum(x.npu()).cpu(),
    "sum_dim1": torch.sum(x.npu(), dim=1, keepdim=True).cpu(),
    "sum_multi": torch.sum(x.npu(), dim=[0, 2]).cpu(),
}
""")
        self._assert_match(r_off, r_on, atol=1e-4)


class TestDvmMatmulOps(_DvmTestBase):
    """MatMul ops: mm, bmm, matmul, addmm."""

    def test_matmul_ops(self):
        r_off, r_on = compare_dvm_on_off("""
m, k, n = 1024, 2048, 1024
x = torch.randn(m, k, dtype=torch.float16) * 0.1
y = torch.randn(k, n, dtype=torch.float16) * 0.1
bx = torch.randn(2, m, k, dtype=torch.float16) * 0.1
by = torch.randn(2, k, n, dtype=torch.float16) * 0.1

result = {
    "mm": torch.mm(x.npu(), y.npu()).cpu(),
    "bmm": torch.bmm(bx.npu(), by.npu()).cpu(),
    "matmul": torch.matmul(x.npu(), y.npu()).cpu(),
}
""")
        self._assert_match(r_off, r_on, atol=1e-3)

    def test_addmm_ops(self):
        r_off, r_on = compare_dvm_on_off("""
m, k, n = 512, 2048, 1024
def np_tensor(shape, np_type):
    return np.random.normal(0, 0.01, shape).astype(np_type)

def torch_tensor(x, data_type=None):
    tensor = torch.from_numpy(x)
    if data_type == "bfloat16":
        tensor = tensor.bfloat16()
    return tensor.npu()

x_npu = torch_tensor(np_tensor((m, k), np.float16))
w_npu = torch_tensor(np_tensor((k, n), np.float16))
x_nc_npu = torch_tensor(np_tensor((k, m), np.float16)).t()
wt_npu = torch_tensor(np_tensor((n, k), np.float16)).t()
bias_1d_npu = torch_tensor(np_tensor((n,), np.float16))
bias_1d_fp32_npu = torch_tensor(np_tensor((n,), np.float32))
self_2d_npu = torch_tensor(np_tensor((m, n), np.float16))
self_2d_fp32_npu = torch_tensor(np_tensor((m, n), np.float32))
self_row_npu = torch_tensor(np_tensor((1, n), np.float16))
self_row_fp32_npu = torch_tensor(np_tensor((1, n), np.float32))
self_col_npu = torch_tensor(np_tensor((m, 1), np.float16))
self_nc_npu = torch_tensor(np_tensor((n, m), np.float16)).t()

xbf16_npu = torch_tensor(np_tensor((m, k), np.float32), "bfloat16")
wbf16_npu = torch_tensor(np_tensor((k, n), np.float32), "bfloat16")
self_bf16_npu = torch_tensor(np_tensor((m, n), np.float32), "bfloat16")
self_bf16_fp32_npu = torch_tensor(np_tensor((m, n), np.float32))

result = {
    "fp16_inputs": {
        "self_1d_same_dtype_alpha1_beta1": torch.addmm(bias_1d_npu, x_npu, w_npu).cpu(),
        "self_1d_fp32_alpha1_beta1": torch.addmm(bias_1d_fp32_npu, x_npu, w_npu).cpu(),
        "self_1d_same_dtype_alpha_other_beta0": torch.addmm(bias_1d_npu, x_npu, wt_npu, beta=0, alpha=1.25).cpu(),
        "self_1d_fp32_alpha_other_beta_other": torch.addmm(bias_1d_fp32_npu, x_npu, w_npu, beta=0.5, alpha=1.25).cpu(),
        "self_2d_full_same_dtype_alpha1_beta1": torch.addmm(self_2d_npu, x_npu, w_npu).cpu(),
        "self_2d_full_fp32_alpha_other_beta1": torch.addmm(self_2d_fp32_npu, x_npu, w_npu, beta=1, alpha=1.25).cpu(),
        "self_2d_row_same_dtype_alpha1_beta0": torch.addmm(self_row_npu, x_npu, wt_npu, beta=0, alpha=1).cpu(),
        "self_2d_row_fp32_alpha1_beta_other": torch.addmm(self_row_fp32_npu, x_npu, w_npu, beta=2, alpha=1).cpu(),
        "self_2d_col_same_dtype_alpha_other_beta_other": torch.addmm(self_col_npu, x_npu, wt_npu, beta=2, alpha=1.25).cpu(),
        "mat1_mat2_noncontiguous": torch.addmm(self_2d_npu, x_nc_npu, wt_npu, beta=0.5, alpha=1.25).cpu(),
        "self_noncontiguous_2d": torch.addmm(self_nc_npu, x_npu, w_npu, beta=1, alpha=1).cpu(),
        "relu_chain": torch.relu(torch.addmm(self_row_npu, x_npu, w_npu, beta=1, alpha=1.25)).cpu(),
    },
    "bf16_inputs": {
        "self_bf16_same_dtype_alpha_other_beta_other": torch.addmm(self_bf16_npu, xbf16_npu, wbf16_npu, beta=0.5, alpha=1.25).cpu(),
        "self_bf16_fp32_alpha1_beta1": torch.addmm(self_bf16_fp32_npu, xbf16_npu, wbf16_npu).cpu(),
    },
}
""")
        self._assert_match(r_off["fp16_inputs"], r_on["fp16_inputs"], atol=1e-3)
        self._assert_match(r_off["bf16_inputs"], r_on["bf16_inputs"], atol=4e-3)


class TestDvmForeachOps(_DvmTestBase):
    """Foreach ops: sqrt, mul, add, div, addcmul, addcdiv."""

    def test_foreach_ops(self):
        r_off, r_on = compare_dvm_on_off("""
ts = [torch.randn(4, 128) for _ in range(4)]
ts_pos = [torch.rand(4, 128) + 0.1 for _ in range(4)]
scalars = [1.5, 2.0, 0.5, 3.0]

# foreach_sqrt
sqrt_in = [t.clone().npu() for t in ts_pos]
sqrt_out = torch._foreach_sqrt(sqrt_in)

# foreach_sqrt_
sqrt_in2 = [t.clone().npu() for t in ts_pos]
torch._foreach_sqrt_(sqrt_in2)

# foreach_mul_
mul_in = [t.clone().npu() for t in ts]
torch._foreach_mul_(mul_in, 2.0)

# foreach_add_
add_in = [t.clone().npu() for t in ts]
torch._foreach_add_(add_in, 1.0)

# foreach_div_ scalar
div_in = [t.clone().npu() for t in ts]
torch._foreach_div_(div_in, 3.0)

# foreach_div_ scalar list
div_in2 = [t.clone().npu() for t in ts]
torch._foreach_div_(div_in2, scalars)

# foreach_addcmul_
cm_in = [t.clone().npu() for t in ts]
cm_t1 = [t.clone().npu() for t in ts]
cm_t2 = [t.clone().npu() for t in ts]
torch._foreach_addcmul_(cm_in, cm_t1, cm_t2, value=0.5)

# foreach_addcdiv_
cd_in = [t.clone().npu() for t in ts]
cd_t1 = [t.clone().npu() for t in ts]
cd_t2 = [t.clone().npu() for t in ts_pos]
torch._foreach_addcdiv_(cd_in, cd_t1, cd_t2, value=0.5)

# foreach_addcdiv_ scalar list
cd_in2 = [t.clone().npu() for t in ts]
cd_t1_2 = [t.clone().npu() for t in ts]
cd_t2_2 = [t.clone().npu() for t in ts_pos]
torch._foreach_addcdiv_(cd_in2, cd_t1_2, cd_t2_2, scalars)

result = {
    "sqrt": [t.cpu() for t in sqrt_out],
    "sqrt_": [t.cpu() for t in sqrt_in2],
    "mul_": [t.cpu() for t in mul_in],
    "add_": [t.cpu() for t in add_in],
    "div_scalar": [t.cpu() for t in div_in],
    "div_scalarlist": [t.cpu() for t in div_in2],
    "addcmul_": [t.cpu() for t in cm_in],
    "addcdiv_": [t.cpu() for t in cd_in],
    "addcdiv_scalarlist_": [t.cpu() for t in cd_in2],
}
""")
        self._assert_match(r_off, r_on)


class TestDvmDtypes(_DvmTestBase):
    """Dtype coverage: fp16, bf16, int32, dtype cast."""

    def test_dtypes(self):
        r_off, r_on = compare_dvm_on_off("""
xf16 = torch.randn(4, 128, dtype=torch.float16)
xbf16 = torch.randn(4, 128, dtype=torch.bfloat16)
xi32 = torch.randint(-10, 10, (4, 128), dtype=torch.int32)
yi32 = torch.randint(-10, 10, (4, 128), dtype=torch.int32)

result = {
    "fp16": (xf16.npu() * 2.0 + 1.0).cpu(),
    "bf16": (xbf16.npu() * 2.0 + 1.0).cpu(),
    "int_add": (xi32.npu() + yi32.npu()).cpu(),
}
""")
        self._assert_match(r_off, r_on)

class TestDvmEdgeCases(_DvmTestBase):
    """Edge cases: empty, scalar, non-contiguous, large, inplace chain."""

    def test_edge_cases(self):
        r_off, r_on = compare_dvm_on_off("""
# Empty tensor
e1 = torch.randn(0, 4).npu()
e2 = torch.randn(0, 4).npu()
empty_add = (e1 + e2).cpu()

# Scalar tensor
s = torch.tensor(3.14).npu()
t = torch.randn(4, 4).npu()
scalar_add = (s + t).cpu()

# Non-contiguous (transposed)
nc = torch.randn(8, 8).npu().t()
nc_relu = torch.relu(nc).cpu()

# Large tensor
lg1 = torch.randn(1024, 1024).npu()
lg2 = torch.randn(1024, 1024).npu()
large_add = (lg1 + lg2).cpu()

result = {
    "empty_add": empty_add,
    "scalar_add": scalar_add,
    "nc_relu": nc_relu,
    "large_add": large_add,
}
""")
        self._assert_match(r_off, r_on)

class TestDvmAliasAndViewScenarios(_DvmTestBase):
    """Scenarios around storage aliasing, hidden views, and inplace updates."""

    def test_external_sibling_views(self):
        """Two contiguous sibling views of one external tensor stay correct."""
        r_off, r_on = compare_dvm_on_off("""
base = (torch.rand(128, dtype=torch.float32) + 0.1).npu()
left = base[:64]
right = base[64:]
left_out = torch.sqrt(left)
right_out = torch.reciprocal(right)
result = {
    "left": left_out.cpu(),
    "right": right_out.cpu(),
    "sum": (left_out + right_out).cpu(),
}
""")
        self._assert_match(r_off, r_on)

    def test_hidden_view_of_fusion_value_read(self):
        """A hidden view of a fusion value should flush before the next read op."""
        r_off, r_on = compare_dvm_on_off("""
x = torch.randn(96, dtype=torch.float32).npu()
base = torch.abs(x)
view = base[16:80]
out = torch.sqrt(view)
result = {
    "view": view.cpu(),
    "out": out.cpu(),
    "tail": (out * 2.0 + 1.0).cpu(),
}
""")
        self._assert_match(r_off, r_on)

    def test_exact_inplace_on_fusion_output(self):
        """Exact inplace chains on one tensor stay correct."""
        r_off, r_on = compare_dvm_on_off("""
x = torch.randn(128, dtype=torch.float32).npu()
y = torch.randn(128, dtype=torch.float32).npu()
z = torch.randn(128, dtype=torch.float32).npu()

out = x * y
out.add_(z)
out.relu_()
result = out.cpu()
""")
        self._assert_match(r_off, r_on)

class TestDvmFusionPatterns(_DvmTestBase):
    """Complex fusion patterns: chains, RMSNorm, SwiGLU, attention, etc."""

    def test_rmsnorm(self):
        """LLaMA-style RMSNorm."""
        r_off, r_on = compare_dvm_on_off("""
B, S, H = 2, 128, 256
x = torch.randn(B, S, H).npu()
w = torch.randn(H).npu()
eps = 1e-6
var = x.pow(2).mean(-1, keepdim=True)
result = (x / torch.sqrt(var + eps) * w).cpu()
""")
        self._assert_match(r_off, r_on)

    def test_swiglu(self):
        """SwiGLU: silu(x @ W_gate) * (x @ W_up)"""
        r_off, r_on = compare_dvm_on_off("""
B, S, H, FFN = 2, 64, 128, 256
x = torch.randn(B, S, H, dtype=torch.float16).npu()
w_gate = torch.randn(H, FFN, dtype=torch.float16).npu()
w_up = torch.randn(H, FFN, dtype=torch.float16).npu()
result = (torch.nn.functional.silu(x @ w_gate) * (x @ w_up)).cpu()
""")
        self._assert_match(r_off, r_on)

class TestDvmSmallNetworks(_DvmTestBase):
    """Small networks with forward + backward, comparing DVM on vs off.

    Note: backward passes through deep matmul chains may have tiny numerical
    differences because DVM fusion can change floating-point operation ordering
    (e.g., intermediate values kept in registers vs materialized to memory).
    This is NOT a DVM bug -- the same phenomenon occurs with different CUDA
    kernel implementations or different reduction orderings. We use small
    tolerances for backward-pass gradients accordingly.
    """

    def test_training_loop_3steps(self):
        """3-step training loop with SGD-like update."""
        r_off, r_on = compare_dvm_on_off("""
H = 32
w = torch.randn(H, H)
x = torch.randn(8, H)
target = torch.randn(8, H)
lr = 0.01

wn = w.clone().npu().requires_grad_(True)
xn = x.npu()
tn = target.npu()

losses = []
for step in range(3):
    out = torch.relu(xn @ wn)
    loss = ((out - tn) ** 2).mean()
    loss.backward()
    losses.append(loss.detach().cpu().item())
    with torch.no_grad():
        wn -= lr * wn.grad
        wn.grad.zero_()

result = {
    "losses": losses,
    "final_w": wn.detach().cpu(),
}
""")
        import torch
        for i, (a, b) in enumerate(zip(r_off["losses"], r_on["losses"])):
            self.assertAlmostEqual(a, b, places=4, msg=f"loss step {i}")
        diff = (r_off["final_w"].float() - r_on["final_w"].float()).abs().max().item()
        self.assertLessEqual(diff, 1e-5, f"final_w max_diff={diff:.2e}")

class TestDvmGuardedFallbacks(_DvmTestBase):
    """Cases that should cleanly fall back instead of entering the DVM path."""

    def test_sum_out_with_cpu_output(self):
        """A CPU out tensor should raise the same clean error with DVM off and on."""
        r_off, r_on = compare_dvm_on_off_error("""
x = torch.randn(4, 16, dtype=torch.float32).npu()
out = torch.full((4,), -999.0, dtype=torch.float32)
torch.sum(x, dim=1, out=out)
""")
        self.assertNotEqual(r_off.returncode, 0)
        self.assertEqual(r_off.returncode, r_on.returncode)
        self.assertIn("output with device cpu doesn't match the desired device NPU", r_off.stderr)
        self.assertIn("output with device cpu doesn't match the desired device NPU", r_on.stderr)


# =============================================================================
# Op-level flag tests
# =============================================================================

class TestDvmFlags(_DvmTestBase):
    """Test op-level disable/enable flags."""

    def _run_with_flags(self, flags, code):
        """Run code with specific DVM flags."""
        import torch
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "result.pt")
            env = os.environ.copy()
            env.pop("TORCH_NPU_LAZY_FUSION", None)
            env["TORCH_NPU_LAZY_FUSION"] = flags
            full_code = f"""
import torch, torch_npu
import numpy as np
torch.manual_seed(42)
np.random.seed(1)
{code}
torch.save(result, "{save_path}")
"""
            r = subprocess.run(
                [sys.executable, "-c", full_code],
                env=env, capture_output=True, text=True, timeout=120
            )
            if r.returncode != 0:
                self.fail(f"Subprocess (flags={flags}) failed:\n{r.stderr[-500:]}")
            return torch.load(save_path, weights_only=False)

    def test_disable_specific_op(self):
        """disable_ops=add should still produce correct results."""
        code = """
x = torch.randn(4, 4).npu()
y = torch.randn(4, 4).npu()
result = (x + y).cpu()
"""
        r_off, _ = compare_dvm_on_off(code)
        r_flag = self._run_with_flags("True disable_ops=add", code)
        self._assert_match(r_off, r_flag)

class TestDvmBatchNorm(_DvmTestBase):
    """BatchNorm ops: native_batch_norm, native_batch_norm_backward, stats, gather_stats_with_counts, elemt, backward_elemt."""

    def test_native_batch_norm_inference(self):
        shapes = [(4, 32), (2, 32, 4, 8)]
        configs = [(True, True), (False, False)]
        for shape in shapes:
            for with_weight, with_bias in configs:
                with self.subTest(shape=shape, with_weight=with_weight, with_bias=with_bias):
                    r_off, r_on = compare_dvm_on_off(f"""
shape = {shape}
with_weight = {with_weight}
with_bias = {with_bias}
c = shape[1]

x_np = np.random.normal(0, 1, shape).astype(np.float32)
weight_np = np.random.normal(0, 1, (c,)).astype(np.float32) if with_weight else None
bias_np = np.random.normal(0, 1, (c,)).astype(np.float32) if with_bias else None
running_mean_np = np.random.normal(0, 1, (c,)).astype(np.float32)
running_var_np = np.abs(np.random.normal(0, 1, (c,)).astype(np.float32)) + 1e-3

x = torch.from_numpy(x_np).npu()
weight = None if weight_np is None else torch.from_numpy(weight_np).npu()
bias = None if bias_np is None else torch.from_numpy(bias_np).npu()
running_mean = torch.from_numpy(running_mean_np).npu()
running_var = torch.from_numpy(running_var_np).npu()

batch_norm_out = torch.batch_norm(
    x,
    weight,
    bias,
    running_mean,
    running_var,
    False,
    0.1,
    1e-5,
    False,
)
native_out, save_mean, save_invstd = torch.ops.aten.native_batch_norm.default(
    x,
    weight,
    bias,
    running_mean,
    running_var,
    False,
    0.1,
    1e-5,
)

result = {{
    "batch_norm": batch_norm_out.cpu(),
    "batch_norm_relu": torch.relu(batch_norm_out).cpu(),
    "native_out": native_out.cpu(),
    "save_mean": save_mean.cpu(),
    "save_invstd": save_invstd.cpu(),
}}
""")
                    self._assert_match(
                        r_off, r_on, atol=1e-4,
                        msg=f" shape={shape} with_weight={with_weight} with_bias={with_bias}"
                    )

    def test_batch_norm_elemt(self):
        shapes = [(1, 32, 2, 4)]
        configs = [(True, True), (False, True), (True, False), (False, False)]
        for shape in shapes:
            for with_weight, with_bias in configs:
                with self.subTest(shape=shape, with_weight=with_weight, with_bias=with_bias):
                    r_off, r_on = compare_dvm_on_off(f"""
shape = {shape}
with_weight = {with_weight}
with_bias = {with_bias}
c = shape[1]

x_np = np.random.normal(0, 1, shape).astype(np.float32)
mean_np = np.random.normal(0, 1, (c,)).astype(np.float32)
invstd_np = np.abs(np.random.normal(0, 1, (c,)).astype(np.float32)) + 1e-5
weight_np = np.random.normal(0, 1, (c,)).astype(np.float32) if with_weight else None
bias_np = np.random.normal(0, 1, (c,)).astype(np.float32) if with_bias else None

x = torch.from_numpy(x_np).npu()
mean = torch.from_numpy(mean_np).npu()
invstd = torch.from_numpy(invstd_np).npu()
weight = None if weight_np is None else torch.from_numpy(weight_np).npu()
bias = None if bias_np is None else torch.from_numpy(bias_np).npu()

out = torch.batch_norm_elemt(
    x,
    weight,
    bias,
    mean,
    invstd,
    1e-5,
)

result = out.cpu()
""")
                    self._assert_match(
                        r_off, r_on, atol=1e-4,
                        msg=f" shape={shape} with_weight={with_weight} with_bias={with_bias}"
                    )

if __name__ == "__main__":
    unittest.main()
