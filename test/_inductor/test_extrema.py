from unittest import mock

import triton.language as tl
from torch.testing._internal.common_utils import (
    TestCase, run_tests, parametrize, instantiate_parametrized_tests,
)
import torch_npu  # noqa: F401
from torch_npu._inductor.runtime import triton_helpers


class TestExtremaPromotion(TestCase):
    @parametrize("dtype", [tl.float16, tl.float32])
    def test_floating_promotion(self, dtype):
        # Exercise the helper's compile-time branch without a device compiler.
        # FP16 must promote; FP32 must also evaluate the condition successfully.
        value = mock.Mock(dtype=dtype)
        with mock.patch.object(tl, "max") as maximum:
            result = triton_helpers._extremum.fn(value, 0, True)
        if dtype == tl.float16:
            value.to.assert_called_once_with(tl.float32)
            expected = value.to.return_value
        else:
            value.to.assert_not_called()
            expected = value
        maximum.assert_called_once_with(expected, 0, propagate_nan=True)
        self.assertIs(result, maximum.return_value)


instantiate_parametrized_tests(TestExtremaPromotion)

if __name__ == "__main__":
    run_tests()
