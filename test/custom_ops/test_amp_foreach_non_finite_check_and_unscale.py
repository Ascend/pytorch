import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestAmpForeachNonFiniteCheckAndUnscale(TestCase):

    @Dtypes(torch.float32, torch.float16)
    def test_grad_scaling_unscale(self, dtype, device="npu"):

        def _clear_float_status():
            float_status = torch.zeros(8).npu()
            result = torch_npu.npu_clear_float_status(float_status)

        def _get_float_status():
            float_status = torch.zeros(8).npu()
            result = torch_npu.npu_get_float_status(float_status)

        inv_scale = torch.full((1,), 0.25, dtype=torch.float, device=device)
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device=device)

        size = 6
        g = torch.full((size, size), 4.0, dtype=dtype, device=device)

        cases = (
            ([g.clone(), g.clone()], False),
            ([g.clone(), g.clone().t()], True),
            ([g.clone(), g.clone()[:, :5]], False),
            ([g.clone()[:, :5], g.clone()[:, :5]], False),
            ([g.clone(), g.clone()], True),
            ([g.clone(), g.clone().t()], False),
        )

        for grads, has_inf in cases:
            found_inf.zero_()
            _clear_float_status()

            if has_inf:
                ginf = g.clone()
                ginf[2, 2].mul_(torch.finfo(dtype).max)
                torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
                _get_float_status()
                self.assertEqual(found_inf, 1.0)
            else:
                torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
                _get_float_status()
                self.assertEqual(found_inf, 0.0)
                for grad in grads:
                    self.assertTrue(torch.allclose(grad, torch.ones_like(grad), atol=1e-7))

        _clear_float_status()

        # Passing lists with mismatched devices or dtypes to a raw
        # _amp_foreach_non_finite_check_and_unscale_ call should raise errors.
        with self.assertRaisesRegex(RuntimeError, r"must have the same dtype"):
            if dtype == torch.float16:
                torch._amp_foreach_non_finite_check_and_unscale_([g.clone(), g.to(dtype=torch.float32)],
                                                                 found_inf,
                                                                 inv_scale)
            else:
                torch._amp_foreach_non_finite_check_and_unscale_([g.clone(), g.to(dtype=torch.float16)],
                                                                 found_inf,
                                                                 inv_scale)


if __name__ == "__main__":
    device_name = torch_npu.npu.get_device_name(0)[:10]
    if device_name in ["Ascend910A"]:
        run_tests()
