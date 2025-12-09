import os
import threading
from ctypes import byref, c_int, c_void_p, CDLL

import torch
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests

import torch_npu


class TestDevice(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 1

    def _check_not_npu(self, device_id=0):
        ascendcl_h = CDLL("libascendcl.so")
        device_id = c_int(device_id)
        activate = c_int(1)
        rc = ascendcl_h.aclrtGetPrimaryCtxState(device_id, c_void_p(), byref(activate))
        if rc != 0:
            raise RuntimeError("call aclrtGetPrimaryCtxState error")
        del ascendcl_h
        self.assertEqual(activate.value, 0)

    def test_current_device(self):
        device = torch.npu.current_device()
        self.assertEqual(device, 0)
        self._check_not_npu()
    
    def test_is_bf16_supported(self):
        torch.npu.is_bf16_supported()
        self._check_not_npu()

    def test_is_support_inf_nan(self):
        torch.npu.utils.is_support_inf_nan()
        self._check_not_npu()


if __name__ == "__main__":
    os.environ["ACL_OP_INIT_MODE"] = "1"
    run_tests()
