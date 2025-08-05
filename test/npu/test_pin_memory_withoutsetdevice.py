import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu


class TestPinMemory(TestCase):

    def test_pin_memory(self):
        # pin_memory -> getPinnedMemoryAllocator -> Initialize -> InitAclops -> aclSetCompileopt
        # now pin_memory hold gil, aclSetCompileopt will fail if we don't release gil
        pin_tensor = torch.tensor((2, 3), pin_memory=True)


if __name__ == '__main__':
    run_tests()
