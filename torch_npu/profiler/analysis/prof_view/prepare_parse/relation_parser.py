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

from ...prof_common_func.constant import Constant, print_error_msg
from ...prof_parse.fwk_cann_relation_parser import FwkCANNRelationParser
from ...prof_view.base_parser import BaseParser


class RelationParser(BaseParser):
    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)

    def run(self, deps_data: dict):
        try:
            kernel_dict = FwkCANNRelationParser(self._profiler_path).get_kernel_dict()
        except Exception:
            print_error_msg("Failed to get acl to npu flow dict.")
            return Constant.FAIL, {}
        return Constant.SUCCESS, kernel_dict
