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

from ..prof_common_func._constant import Constant
from ..prof_common_func._task_manager import ConcurrentMode

__all__ = []


class ParserDepsConfig:
    COMMON_CONFIG = {
        Constant.TORCH_OP_PARSER: {Constant.MODE: ConcurrentMode.PTHREAD, Constant.DEPS: []},
        Constant.TASK_QUEUE_PARSER: {Constant.MODE: ConcurrentMode.PTHREAD, Constant.DEPS: []},
        Constant.TRACE_PRE_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS,
                                    Constant.DEPS: [Constant.TORCH_OP_PARSER, Constant.TASK_QUEUE_PARSER]},
        Constant.TREE_BUILD_PARSER: {Constant.MODE: ConcurrentMode.PTHREAD,
                                     Constant.DEPS: [Constant.TORCH_OP_PARSER, Constant.TASK_QUEUE_PARSER]},
        Constant.CANN_EXPORT_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS, Constant.DEPS: []},
        Constant.CANN_TIMELINE_PARSER: {Constant.MODE: ConcurrentMode.NON_BLOCKING | ConcurrentMode.PTHREAD,
                                        Constant.DEPS: []},
        Constant.RELATION_PARSER: {Constant.MODE: ConcurrentMode.PTHREAD,
                                   Constant.DEPS: [Constant.TASK_QUEUE_PARSER, Constant.CANN_TIMELINE_PARSER]},
        Constant.CANN_ANALYZE_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS,
                                       Constant.DEPS: [Constant.CANN_TIMELINE_PARSER]},
        Constant.OPERATOR_VIEW_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS,
                                        Constant.DEPS: [Constant.TREE_BUILD_PARSER, Constant.CANN_TIMELINE_PARSER,
                                                        Constant.RELATION_PARSER, Constant.TORCH_OP_PARSER]},
        Constant.TRACE_VIEW_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS,
                                     Constant.DEPS: [Constant.TREE_BUILD_PARSER, Constant.TRACE_PRE_PARSER,
                                                     Constant.CANN_TIMELINE_PARSER, Constant.TASK_QUEUE_PARSER,
                                                     Constant.TORCH_OP_PARSER]},
        Constant.KERNEL_VIEW_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS,
                                      Constant.DEPS: [Constant.TREE_BUILD_PARSER, Constant.CANN_EXPORT_PARSER,
                                                      Constant.RELATION_PARSER]},
        Constant.TRACE_STEP_TIME_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS,
                                          Constant.DEPS: [Constant.TREE_BUILD_PARSER, Constant.CANN_TIMELINE_PARSER,
                                                          Constant.RELATION_PARSER, Constant.TORCH_OP_PARSER]},
        Constant.MEMORY_VIEW_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS,
                                      Constant.DEPS: [Constant.CANN_EXPORT_PARSER, Constant.MEMORY_PREPARE]},
        Constant.INTEGRATE_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS,
                                    Constant.DEPS: [Constant.CANN_EXPORT_PARSER]},
        Constant.COMMUNICATION_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS,
                                        Constant.DEPS: [Constant.TREE_BUILD_PARSER, Constant.CANN_ANALYZE_PARSER,
                                                        Constant.RELATION_PARSER]},
        Constant.STACK_VIEW_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS,
                                     Constant.DEPS: [Constant.TREE_BUILD_PARSER, Constant.CANN_TIMELINE_PARSER,
                                                     Constant.TASK_QUEUE_PARSER, Constant.TORCH_OP_PARSER]},
        Constant.MEMORY_PREPARE: {Constant.MODE: ConcurrentMode.PTHREAD,
                                  Constant.DEPS: [Constant.TREE_BUILD_PARSER, Constant.TASK_QUEUE_PARSER,
                                                  Constant.TORCH_OP_PARSER]},
        Constant.DB_PRE_PARSER: {Constant.MODE: ConcurrentMode.PTHREAD,
                                 Constant.DEPS: [Constant.TORCH_OP_PARSER, Constant.TASK_QUEUE_PARSER]},
        Constant.DB_PARSER: {Constant.MODE: ConcurrentMode.PTHREAD,
                             Constant.DEPS: [Constant.CANN_EXPORT_PARSER, Constant.MEMORY_PREPARE,
                                             Constant.TREE_BUILD_PARSER, Constant.CANN_ANALYZE_PARSER,
                                             Constant.DB_PRE_PARSER]},
        Constant.MEMORY_TIMELINE_PARSER: {}
    }

    ONLY_FWK_CONFIG = {
        Constant.TORCH_OP_PARSER: {Constant.MODE: ConcurrentMode.PTHREAD, Constant.DEPS: []},
        Constant.TASK_QUEUE_PARSER: {Constant.MODE: ConcurrentMode.PTHREAD, Constant.DEPS: []},
        Constant.OPERATOR_VIEW_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS, Constant.DEPS: [Constant.TORCH_OP_PARSER]},
        Constant.TRACE_VIEW_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS, Constant.DEPS: [Constant.TASK_QUEUE_PARSER, Constant.TORCH_OP_PARSER]},
        Constant.MEMORY_VIEW_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS, Constant.DEPS: [Constant.MEMORY_PREPARE]},
        Constant.STACK_VIEW_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS, Constant.DEPS: [Constant.TORCH_OP_PARSER, Constant.TASK_QUEUE_PARSER]},
        Constant.CANN_EXPORT_PARSER: {Constant.MODE: ConcurrentMode.SUB_PROCESS, Constant.DEPS: []},
        Constant.DB_PRE_PARSER: {Constant.MODE: ConcurrentMode.PTHREAD,
                                 Constant.DEPS: [Constant.TORCH_OP_PARSER, Constant.TASK_QUEUE_PARSER]},
        Constant.MEMORY_PREPARE: {Constant.MODE: ConcurrentMode.PTHREAD,
                                  Constant.DEPS: [Constant.TASK_QUEUE_PARSER, Constant.TORCH_OP_PARSER]},
        Constant.DB_PARSER: {Constant.MODE: ConcurrentMode.PTHREAD, Constant.DEPS: [Constant.DB_PRE_PARSER, Constant.MEMORY_PREPARE]},
        Constant.MEMORY_TIMELINE_PARSER: {}
    }
