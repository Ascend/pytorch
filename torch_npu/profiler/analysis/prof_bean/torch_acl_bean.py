# Copyright (c) 2023, Huawei Technologies.
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

class TorchAclBean:
    def __init__(self, data: dict):
        self._origin_data = data

    @property
    def acl_start_time(self):
        return self._origin_data.get("acl_start_time")

    @property
    def op_name(self):
        return self._origin_data.get("op_name")

    @property
    def torch_op_start_time(self):
        return self._origin_data.get("torch_op_start_time")

    @property
    def torch_op_tid(self):
        return self._origin_data.get("torch_op_tid")

    @property
    def torch_op_pid(self):
        return self._origin_data.get("torch_op_pid")

    @property
    def npu_kernel_list(self):
        return [data.replace("-", "_") for data in self._origin_data.get("npu_kernel_list")]
