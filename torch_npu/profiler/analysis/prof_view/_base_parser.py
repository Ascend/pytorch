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

from abc import ABC

from ..prof_common_func._constant import Constant
from ..prof_common_func._path_manager import ProfilerPathManager
from ..prof_common_func._task_manager import ConcurrentMode, ConcurrentTask
from ..prof_config._parser_deps_config import ParserDepsConfig

__all__ = []


class BaseParser(ConcurrentTask, ABC):
    def __init__(self, name: str, param_dict: dict):
        self._param_dict = param_dict
        self._profiler_path = None
        self._output_path = None
        deps, mode = self._init_param(name)
        self._export_type = param_dict.get(Constant.EXPORT_TYPE, [])
        super(BaseParser, self).__init__(name, deps, mode)

    def _init_param(self, name: str) -> any:
        self._profiler_path = self._param_dict.get("profiler_path")
        self._output_path = self._param_dict.get("output_path")
        if ProfilerPathManager.get_cann_path(self._profiler_path):
            config = ParserDepsConfig.COMMON_CONFIG.get(name, {})
        else:
            config = ParserDepsConfig.ONLY_FWK_CONFIG.get(name, {})
        mode = config.get(Constant.MODE, ConcurrentMode.SUB_PROCESS)
        deps_parser = config.get(Constant.DEPS, [])
        return deps_parser, mode
