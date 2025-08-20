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

import os

from ...prof_common_func._constant import Constant
from ...prof_common_func._file_manager import FileManager
from ...prof_common_func._log import ProfilerLogger
from ...prof_parse._fwk_file_parser import FwkFileParser
from ...prof_view._base_parser import BaseParser
from ...prof_common_func._file_tag import FileTag

__all__ = []


class TracePreParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "TracePreParser")
        self.logger = ProfilerLogger.get_instance()
        self.logger.info("TracePreParser start.")
        try:
            torch_op_data = deps_data.get(Constant.TORCH_OP_PARSER, [])
            task_queue_data = deps_data.get(Constant.TASK_QUEUE_PARSER, {})
            enqueue_data = task_queue_data.get(Constant.ENQUEUE_DATA, [])
            dequeue_data = task_queue_data.get(Constant.DEQUEUE_DATA, [])
            fwk_trace_data = FwkFileParser(self._profiler_path).get_fwk_trace_data(
                torch_op_data, enqueue_data, dequeue_data)
            trace_file_path = os.path.join(self._output_path, Constant.TRACE_VIEW) if os.path.isdir(
                self._output_path) else self._output_path
            FileManager.create_prepare_trace_json_by_path(trace_file_path, fwk_trace_data)
        except Exception as e:
            self.logger.error("Failed to create prepare trace json, error: %s", str(e), exc_info=True)
            return Constant.FAIL, None
        self.logger.info("TracePreParser finish.")
        return Constant.SUCCESS, None


class TreeBuildParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)
        ProfilerLogger.init(self._profiler_path, "TracePreParser")
        self.logger = ProfilerLogger.get_instance()

    def run(self, deps_data: dict):
        self.logger.info("TreeBuildParser start.")
        try:
            enqueue_data = deps_data.get(Constant.TASK_QUEUE_PARSER, {}).get(Constant.ENQUEUE_DATA, [])
            torch_op_data = deps_data.get(Constant.TORCH_OP_PARSER, [])
            torch_op_node = FwkFileParser(self._profiler_path).get_torch_op_tree_node(torch_op_data, enqueue_data)
        except Exception as e:
            self.logger.error("Failed to build torch op tree, error: %s", str(e), exc_info=True)
            return Constant.FAIL, []
        self.logger.info("TreeBuildParser finish.")
        return Constant.SUCCESS, torch_op_node


class TaskQueueParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "TaskQueueParser")
        self.logger = ProfilerLogger.get_instance()
        self.logger.info("TaskQueueParser start.")
        try:
            enqueue_data, dequeue_data = FwkFileParser(self._profiler_path).get_task_queue_data()
        except Exception as e:
            self.logger.error("Failed to get task queue data, error: %s", str(e), exc_info=True)
            return Constant.FAIL, {}
        self.logger.info("TaskQueueParser finish.")
        return Constant.SUCCESS, {Constant.ENQUEUE_DATA: enqueue_data, Constant.DEQUEUE_DATA: dequeue_data}


class TorchOpParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "TorchOpParser")
        self.logger = ProfilerLogger.get_instance()
        self.logger.info("TorchOpParser start.")
        try:
            torch_op_data = FwkFileParser(self._profiler_path).get_file_data_by_tag(FileTag.TORCH_OP)
        except Exception as e:
            self.logger.error("Failed to get torch op tree, error: %s", str(e), exc_info=True)
            return Constant.FAIL, []
        self.logger.info("TorchOpParser finish.")
        return Constant.SUCCESS, torch_op_data


class DbPreParser(BaseParser):

    def __init__(self, name: str, param_dict: dict):
        super().__init__(name, param_dict)

    def run(self, deps_data: dict):
        ProfilerLogger.init(self._profiler_path, "DbPreParser")
        self.logger = ProfilerLogger.get_instance()
        self.logger.info("DbPreParser start.")
        try:
            torch_op_data = deps_data.get(Constant.TORCH_OP_PARSER, [])
            task_queue_data = deps_data.get(Constant.TASK_QUEUE_PARSER, {})
            enqueue_data = task_queue_data.get(Constant.ENQUEUE_DATA, [])
            dequeue_data = task_queue_data.get(Constant.DEQUEUE_DATA, [])
            fwk_db_data = FwkFileParser(self._profiler_path).get_fwk_api(
                torch_op_data, enqueue_data, dequeue_data)
        except Exception as e:
            self.logger.error("Failed to create prepare db data, error: %s", str(e), exc_info=True)
            return Constant.FAIL, None
        self.logger.info("DbPreParser finish.")
        return Constant.SUCCESS, fwk_db_data
