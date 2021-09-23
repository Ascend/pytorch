
# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
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

"""
This global flag control mm and bmm use ND format to compute, if the flag is True,
we use ND format for mm and bmm in Linear module

useage:

option = {}
option["MM_BMM_ND_ENABLE"] = "enable"
torch.npu.set_option(option)

Default: False

"""
_MM_BMM_ND_ENABLE = False
def set_mm_bmm_format_nd(val):
    global _MM_BMM_ND_ENABLE
    _MM_BMM_ND_ENABLE = val

def get_mm_bmm_format_nd():
    return _MM_BMM_ND_ENABLE