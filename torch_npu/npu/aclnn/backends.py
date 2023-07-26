# Copyright (c) 2023, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import warnings
import torch_npu._C

from torch.backends import ContextProp, PropModule


def version():
    """Currently, the ACLNN version is not available and does not support it. 
    By default, it returns None.
    """
    warnings.warn("torch_npu.npu.aclnn.version isn't implemented!")
    return None


def _set_allow_conv_hf32(_enabled: bool):
    r"""Set the device supports conv operation hf32.
    Args:
        Switch for hf32.
    """
    option = {"ALLOW_CONV_HF32": "enable" if _enabled else "disable"}
    torch_npu._C._npu_setOption(option)


def _get_allow_conv_hf32() -> bool:
    r"""Return the device supports conv operation hf32 is enabled or not.
    """
    hf32_value = torch_npu._C._npu_getOption("ALLOW_CONV_HF32")
    return (hf32_value is None) or (hf32_value.decode() == "") or (hf32_value.decode() == "enable")


class AclnnModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    allow_hf32 = ContextProp(_get_allow_conv_hf32, _set_allow_conv_hf32)


sys.modules[__name__] = AclnnModule(sys.modules[__name__], __name__)
allow_hf32: bool
