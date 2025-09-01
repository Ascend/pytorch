import os
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils
import torch_npu

os.environ["INDUCTOR_ASCEND_CHECK_ACCURACY"] = "1"


class TestCheckAccuracy(TestUtils):
    def test_check_accuracy_1(self):
        count_data_dump = 0
        count_check_accuracy = 0
        
        def run(x, y):
            return F.relu(x) - y

        from torch_npu._inductor.npu_triton_heuristics import NPUCachingAutotuner
        src_data_dump = NPUCachingAutotuner.data_dump

        def wrap_data_dump(self, *args, **kwargs):
            status = src_data_dump(self, *args, **kwargs)
            if status:
                nonlocal count_data_dump
                count_data_dump += 1
            return status
        
        src_check_accuracy = NPUCachingAutotuner.check_accuracy

        def wrap_check_accuracy(self, *args, **kwargs):
            status = src_check_accuracy(self, *args, **kwargs)
            if status:
                nonlocal count_check_accuracy
                count_check_accuracy += 1
            return status

        x = torch.randn(10).npu()
        y = torch.randn(10).npu()
        g = run(x, y)

        run = torch.compile(run)
        # compile warmup
        _ = run(x, y)

        with patch.object(NPUCachingAutotuner, "data_dump", wrap_data_dump), \
            patch.object(NPUCachingAutotuner, "check_accuracy", wrap_check_accuracy):
            self.assertTrue(torch_npu._inductor.config.dump_fx_graph)
            self.assertTrue(torch_npu._inductor.config.check_accuracy)
            
            # Try run custom path and make sure no data_dump and check_accuracy is invoked.
            torch_npu._inductor.config.dump_fx_graph = False
            torch_npu._inductor.config.check_accuracy = False
            z = run(x, y)
            self.assertEqual(count_data_dump, 0)
            self.assertEqual(count_check_accuracy, 0)
            self.assertEqual(z, g)

            torch_npu._inductor.config.dump_fx_graph = True
            torch_npu._inductor.config.check_accuracy = True
            z = run(x, y)
            self.assertEqual(count_data_dump, 1)
            self.assertEqual(count_check_accuracy, 1)
            self.assertEqual(z, g)
        

if __name__ == "__main__":
    run_tests()