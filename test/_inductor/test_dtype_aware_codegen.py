# Backend selection is resolved when torch_npu._inductor is imported, so this
# file pins the TE backend via process env before those imports. The vendor CI
# runs each test file in its own process; anything importing this module
# in-process inherits the pin.
import functools
import os
import unittest

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._dynamo.utils import disable_cache_limit
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

try:
    import torch_npu  # noqa: F401
    import torch_npu._inductor  # noqa: F401
    IS_NPU_AVAILABLE = torch.npu.is_available()
except (ImportError, RuntimeError):
    IS_NPU_AVAILABLE = False

DEVICE = "npu"

# Ops on the upstream triton lowering for f16/bf16 with the fp32 upcast: the
# routing additions, erf/log1p (pre-existing GENERATE_LIST members on the same
# lowering), and rsqrt (maybe_upcast_float32 added in the same fix). Each
# entry keeps its own lambda so every op compiles in a distinct dynamo frame.
UPCAST_OPS = {
    "acos": lambda a: torch.acos(a),
    "acosh": lambda a: torch.acosh(a),
    "asin": lambda a: torch.asin(a),
    "asinh": lambda a: torch.asinh(a),
    "atan": lambda a: torch.atan(a),
    "atanh": lambda a: torch.atanh(a),
    "atan2": lambda a, b: torch.atan2(a, b),
    "ceil": lambda a: torch.ceil(a),
    "copysign": lambda a, b: torch.copysign(a, b),
    "cosh": lambda a: torch.cosh(a),
    "erf": lambda a: torch.erf(a),
    "erfinv": lambda a: torch.erfinv(a),
    "expm1": lambda a: torch.expm1(a),
    "floor": lambda a: torch.floor(a),
    "fmod": lambda a, b: torch.fmod(a, b),
    "isinf": lambda a: torch.isinf(a),
    "isnan": lambda a: torch.isnan(a),
    "lgamma": lambda a: torch.lgamma(a),
    "log10": lambda a: torch.log10(a),
    "log1p": lambda a: torch.log1p(a),
    "nextafter": lambda a, b: torch.nextafter(a, b),
    "rsqrt": lambda a: torch.rsqrt(a),
    "sinh": lambda a: torch.sinh(a),
    "trunc": lambda a: torch.trunc(a),
}

# Ops whose math domain differs from randn support. The +1.5 offset keeps
# the whole group in-domain; acosh requires inputs >= 1.
DOMAIN_CLAMP = ("acos", "asin", "atan", "atanh", "erfinv")
DOMAIN_POS = ("acosh", "lgamma", "log10", "log1p", "rsqrt")

UPCAST_SHAPE = (256, 256)


def make_upcast_inputs(op_name, shape, dtype):
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    if op_name in DOMAIN_CLAMP:
        x = x.clamp(-0.95, 0.95)
    elif op_name in DOMAIN_POS:
        x = x.abs() + 1.5
    elif op_name == "isinf":
        x = x.clone()
        x[0, 0] = float("inf")
    if op_name in ("atan2", "copysign", "nextafter"):
        return (x, torch.randn(shape, device=DEVICE, dtype=dtype))
    if op_name == "fmod":
        y = torch.floor(torch.randn(shape, device=DEVICE, dtype=dtype) * 2) + 0.5
        return (x, y)
    return (x,)


@functools.cache
def _nextafter_triton_compiles():
    # The probe must mirror the real test's compile form: a smaller or
    # simplified compile can succeed where the real kernel fails.
    try:
        args = make_upcast_inputs("nextafter", UPCAST_SHAPE, torch.float16)
        compiled = torch.compile(UPCAST_OPS["nextafter"], backend="inductor", dynamic=False)
        compiled(*args)
    except Exception:
        return False
    return True


@unittest.skipUnless(IS_NPU_AVAILABLE, "requires torch_npu and an NPU device")
class TestDtypeAwareUpcast(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("op_name", list(UPCAST_OPS))
    def test_dtype_aware_upcast_contract(self, op_name, dtype):
        if op_name == "nextafter" and not _nextafter_triton_compiles():
            self.skipTest(
                "nextafter's triton kernel needs a bishengir compare-mode "
                "mapping missing up to CANN 9.2.0-beta.2; unskip by upgrading "
                "the CI image to >= 9.2.0 release"
            )
        args = make_upcast_inputs(op_name, UPCAST_SHAPE, dtype)
        fn = UPCAST_OPS[op_name]
        compiled = torch.compile(fn, backend="inductor", dynamic=False)
        out, code = run_and_get_code(compiled, *args)
        self.assertTrue(
            any(".to(tl.float32)" in c for c in code),
            f"{op_name} {dtype}: dtype-aware upcast cast missing in generated code",
        )
        self.assertEqual(fn(*args), out, atol=1e-2, rtol=1e-2)


REDUCTIONS = {
    "sum": torch.sum,
    "prod": torch.prod,
    "max": torch.max,
    "min": torch.min,
}


@unittest.skipUnless(IS_NPU_AVAILABLE, "requires torch_npu and an NPU device")
class TestLowPrecisionReduction(TestCase):
    # Port of the upstream test_low_precision_reduction contract: with the
    # global upcast flag off, low-precision full reductions must still
    # upcast to fp32 in the generated kernel and stay close to eager. The
    # input is one tensor repeated four times (reduced values are x**4,
    # all positive), matching the upstream test's input pattern.
    # argmax/argmin from the upstream parametrization are excluded: their
    # low-precision lowering is a separate vendor gap, not covered by
    # this fix.
    @disable_cache_limit()
    @parametrize("input_shape", [(32, 32), (32, 128), (256, 32)])
    @parametrize("reduction_name", ["sum", "prod", "max", "min"])
    @parametrize("input_dtype", [torch.float16, torch.bfloat16])
    @config.patch("triton.use_block_ptr", True)
    def test_low_precision_reduction(self, input_shape, reduction_name, input_dtype):
        x = torch.rand(input_shape, device=DEVICE, dtype=input_dtype)

        def fn(a, b, c, d):
            return REDUCTIONS[reduction_name](a * b * c * d)

        with config.patch("triton.codegen_upcast_to_fp32", False):
            out, code = run_and_get_code(
                torch.compile(fn, backend="inductor", dynamic=False), x, x, x, x
            )
        self.assertTrue(
            ".to(tl.float32)" in "\n".join(code),
            f"{reduction_name} {input_dtype}: reduction upcast missing in generated code",
        )
        self.assertEqual(fn(x, x, x, x), out, atol=1e-2, rtol=2e-2)


@unittest.skipUnless(IS_NPU_AVAILABLE, "requires torch_npu and an NPU device")
class TestDtypePromotionOverloads(TestCase):
    # extern fallback has no type promotion; the decomp table gating makes
    # integral/bool inputs produce float32 output matching eager promotion.

    def test_erf_bool_output_promotes_to_float32(self):
        x = torch.tensor([True, False, True], device=DEVICE)

        def fn(a):
            return torch.erf(a)

        out = torch.compile(fn, backend="inductor", dynamic=False)(x)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(fn(x), out)

    def test_expm1_bool_output_promotes_to_float32(self):
        x = torch.tensor([True, False, True], device=DEVICE)

        def fn(a):
            return torch.expm1(a)

        out = torch.compile(fn, backend="inductor", dynamic=False)(x)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(fn(x), out)

    def test_expm1_int64_output_promotes_to_float32(self):
        x = torch.tensor([1, 2, 3], device=DEVICE, dtype=torch.int64)

        def fn(a):
            return torch.expm1(a)

        out = torch.compile(fn, backend="inductor", dynamic=False)(x)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(fn(x), out)

    # Graph nodes can carry any overload; the routing must cover every
    # overload of the op packet, not just .default.
    def test_copysign_out_overload_compiles(self):
        x = torch.randn((64, 64), device=DEVICE, dtype=torch.float16)
        y = torch.randn((64, 64), device=DEVICE, dtype=torch.float16)

        def fn(a, b):
            out = torch.empty_like(a)
            torch.ops.aten.copysign.out(a, b, out=out)
            return out

        out, code = run_and_get_code(
            torch.compile(fn, backend="inductor", dynamic=False), x, y
        )
        self.assertTrue(
            any(".to(tl.float32)" in c for c in code),
            "copysign.out: dtype-aware upcast cast missing in generated code",
        )
        self.assertEqual(torch.copysign(x, y), out, atol=1e-2, rtol=1e-2)


instantiate_parametrized_tests(TestDtypeAwareUpcast)
instantiate_parametrized_tests(TestLowPrecisionReduction)
instantiate_parametrized_tests(TestDtypePromotionOverloads)

if __name__ == "__main__":
    run_tests()
