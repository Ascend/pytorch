from torch.testing._internal.opinfo.core import OpInfo
from torch_npu.testing.npu_testing_utils import update_skip_list, get_decorators


def apply_test_patchs():
    update_skip_list()
    OpInfo.get_decorators = get_decorators

#apply test_ops related patch
apply_test_patchs()
