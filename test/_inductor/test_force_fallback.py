import os
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils
import torch_npu

os.environ["INDUCTOR_ASCEND_DUMP_FX_GRAPH"] = "1"


class TestForceFallback(TestUtils):
    def test_case1(self):
        op_list = []

        def opoverload_call(self, /, *args, **kwargs):
            op_list.append(str(self))
            return self._op(*args, **kwargs)
        
        def run(x, y):
            return F.relu(x) + y
        
        x = torch.randn(10).npu()
        y = torch.randn(10).npu()
        g = run(x, y)

        run = torch.compile(run)
        # compile warmup
        _ = run(x, y)

        with patch.object(torch._ops.OpOverload, "__call__", opoverload_call):
            op_list.clear()
            z = run(x, y)
            self.assertTrue(len(op_list) == 0)
            self.assertEqual(z, g)

            op_list.clear()
            torch_npu._inductor.config.force_fallback_kernel_id = [0]
            z = run(x, y)
            self.assertTrue("aten.relu.default" in op_list)
            self.assertTrue("aten.add.Tensor" in op_list)
            self.assertEqual(z, g)

            op_list.clear()
            torch_npu._inductor.config.force_fallback_kernel_id = 'all'
            z = run(x, y)
            self.assertTrue("aten.relu.default" in op_list)
            self.assertTrue("aten.add.Tensor" in op_list)
            self.assertEqual(z, g)
        
        # reset
        torch_npu._inductor.config.force_fallback_kernel_id = []


if __name__ == "__main__":
    run_tests()