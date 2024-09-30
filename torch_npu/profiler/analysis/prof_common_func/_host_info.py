# Copyright (c) 2024, Huawei Technologies.
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

import socket
from torch_npu._C._profiler import _get_host_uid

__all__ = []


def get_host_info() -> dict:
    host_name = socket.gethostname()
    host_uid = str(_get_host_uid())
    return {
        'host_name': host_name,
        'host_uid': host_uid
    }
