from typing import Any, Dict, List, Tuple, Union
import functools
import math
import numpy as np
from torch._functorch.aot_autograd import aot_module_simplified

import torch
from torch.library import Library, impl
from torch._subclasses.fake_tensor import FakeTensorMode

# include meta infer
import meta
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

fake_mode = FakeTensorMode()


class TestPromptFlashAttention(TestCase):
    def testPromptFlashAttention(self):
        with fake_mode:
            q = torch.randn(1, 40, 1, 128, dtype=torch.float16).npu()
            k = torch.randn(1, 40, 1, 128, dtype=torch.float16).npu()
            v = torch.randn(1, 40, 1, 128, dtype=torch.float16).npu()
            q.requires_grad = True
            k.requires_grad = True
            v.requires_grad = True
            res = torch.ops.npu.npu_prompt_flash_attention(q, k, v)

            print("q.shape: ", q.shape)
            print("res.shape: ", res.shape)
            self.assertTrue(q.shape == res.shape)


if __name__ == "__main__":
    run_tests()
