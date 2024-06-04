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

from ..prof_common_func.constant import Constant
from ..prof_view.cann_parse.cann_analyze import CANNAnalyzeParser
from ..prof_view.cann_parse.cann_export import CANNExportParser, CANNTimelineParser
from ..prof_view.memory_prepare_parser import MemoryPrepareParser
from ..prof_view.prepare_parse.fwk_pre_parser import TracePreParser, TreeBuildParser
from ..prof_view.kernel_view_parser import KernelViewParser
from ..prof_view.operator_view_parser import OperatorViewParser
from ..prof_view.prepare_parse.relation_parser import RelationParser
from ..prof_view.stack_view_parser import StackViewParser
from ..prof_view.trace_step_time_parser import TraceStepTimeParser
from ..prof_view.trace_view_parser import TraceViewParser
from ..prof_view.memory_view_parser import MemoryViewParser
from ..prof_view.integrate_parser import IntegrateParser
from ..prof_view.communication_parser import CommunicationParser
from ..prof_view.prof_db_parse.fwk_api_db_parser import FwkApiDbParser
from ..prof_view.prof_db_parse.memory_db_parser import MemoryDbParser
from ..prof_view.prof_db_parse.db_parser import DbParser
from ..prof_view.prof_db_parse.step_info_db_parser import StepInfoDbParser
from ..prof_view.prof_db_parse.communication_db_parser import CommunicationDbParser
from ..prof_view.prof_db_parse.trace_step_time_db_parser import TraceStepTimeDbParser

__all__ = []


class ParserConfig:
    LEVEL_NONE_CONFIG = {
        Constant.Text: {
            Constant.TENSORBOARD_TRACE_HANDLER: [
                TracePreParser,
                TreeBuildParser,
                CANNExportParser,
                CANNTimelineParser,
                RelationParser,
                MemoryPrepareParser,
                CANNAnalyzeParser,
                OperatorViewParser,
                TraceViewParser,
                MemoryViewParser,
                IntegrateParser,
            ]
        },
        Constant.Db: {
            Constant.TENSORBOARD_TRACE_HANDLER: [
                CANNExportParser,
                DbParser,
                CANNTimelineParser,
                CANNAnalyzeParser,
                FwkApiDbParser,
                TreeBuildParser,
                MemoryPrepareParser,
                MemoryDbParser
            ]
        }
    }

    COMMON_CONFIG = {
        Constant.Text: {
            Constant.TENSORBOARD_TRACE_HANDLER: [
                TracePreParser,
                TreeBuildParser,
                CANNExportParser,
                CANNTimelineParser,
                RelationParser,
                MemoryPrepareParser,
                CANNAnalyzeParser,
                OperatorViewParser,
                TraceViewParser,
                KernelViewParser,
                TraceStepTimeParser,
                MemoryViewParser,
                IntegrateParser,
                CommunicationParser
            ],
            Constant.EXPORT_CHROME_TRACE: [TracePreParser, TreeBuildParser, CANNExportParser, CANNTimelineParser,
                                           TraceViewParser],
            Constant.EXPORT_STACK: [TreeBuildParser, CANNExportParser, CANNTimelineParser, StackViewParser]
        },
        Constant.Db: {
            Constant.TENSORBOARD_TRACE_HANDLER: [
                CANNExportParser,
                DbParser,
                CANNTimelineParser,
                CANNAnalyzeParser,
                FwkApiDbParser,
                TreeBuildParser,
                MemoryPrepareParser,
                MemoryDbParser,
                StepInfoDbParser,
                CommunicationDbParser,
                TraceStepTimeDbParser
            ]
        }
    }

    ONLY_FWK_CONFIG = {
        Constant.Text: {
            Constant.TENSORBOARD_TRACE_HANDLER: [OperatorViewParser, TraceViewParser, MemoryViewParser],
            Constant.EXPORT_CHROME_TRACE: [TraceViewParser],
            Constant.EXPORT_STACK: [StackViewParser]
        },
        Constant.Db: {
            Constant.TENSORBOARD_TRACE_HANDLER: [CANNExportParser, DbParser, FwkApiDbParser, MemoryDbParser]
        }
    }

    PARSER_NAME_MAP = {
        # text parser
        TracePreParser: Constant.TRACE_PRE_PARSER,
        TreeBuildParser: Constant.TREE_BUILD_PARSER,
        CANNExportParser: Constant.CANN_EXPORT_PARSER,
        CANNTimelineParser: Constant.CANN_TIMELINE_PARSER,
        CANNAnalyzeParser: Constant.CANN_ANALYZE_PARSER,
        OperatorViewParser: Constant.OPERATOR_VIEW_PARSER,
        TraceViewParser: Constant.TRACE_VIEW_PARSER,
        KernelViewParser: Constant.KERNEL_VIEW_PARSER,
        TraceStepTimeParser: Constant.TRACE_STEP_TIME_PARSER,
        MemoryViewParser: Constant.MEMORY_VIEW_PARSER,
        IntegrateParser: Constant.INTEGRATE_PARSER,
        CommunicationParser: Constant.COMMUNICATION_PARSER,
        RelationParser: Constant.RELATION_PARSER,
        StackViewParser: Constant.STACK_VIEW_PARSER,
        MemoryPrepareParser: Constant.MEMORY_PREPARE,

        # db parser
        DbParser: Constant.DB_PARSER,
        FwkApiDbParser: Constant.FWK_API_DB_PARSER,
        MemoryDbParser: Constant.MEMORY_DB_PARSER,
        StepInfoDbParser: Constant.STEP_INFO_DB_PARSER,
        CommunicationDbParser: Constant.COMMUNICATION_DB_PARSER,
        TraceStepTimeDbParser: Constant.TRACE_STEP_TIME_DB_PARSER
    }
