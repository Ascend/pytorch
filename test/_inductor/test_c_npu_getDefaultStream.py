import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


class TestNpuSetDeviceInGraph(TestUtils):

    @parametrize("shape", [(2, 2)])
    @parametrize("dtype", ["float32"])
    def test_npu_set_device_in_graph(self, shape, dtype):
        def op_calc(x, y):
            x = x * y
            # 保留你原来 graph 内的 NPU 调用
            torch_npu._C._npu_getDefaultStream(0)
            out = x + y
            return out

        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)

        compiled_fn = torch.compile(op_calc, backend="inductor")
        out, codes = run_and_get_code(compiled_fn, x, y)

        # 基本 sanity check：确保 Inductor kernel 生成
        self.assertTrue(len(codes) > 0)
        self.assertIn("triton", codes[0])


instantiate_parametrized_tests(TestNpuSetDeviceInGraph)


if __name__ == "__main__":
    run_tests()