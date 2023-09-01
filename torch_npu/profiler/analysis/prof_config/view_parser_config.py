from ..prof_common_func.constant import Constant
from ..prof_view.kernel_view_parser import KernelViewParser
from ..prof_view.operator_view_parser import OperatorViewParser
from ..prof_view.trace_view_parser import TraceViewParser
from ..prof_view.memory_view_parser import MemoryViewParser
from ..prof_view.integrate_parser import IntegrateParser
from ..prof_view.communication_parser import CommunicationParser


class ViewParserConfig(object):
    CONFIG_DICT = {
        Constant.TENSORBOARD_TRACE_HANDLER: [OperatorViewParser, TraceViewParser, KernelViewParser,
                                             MemoryViewParser, IntegrateParser, CommunicationParser],
        Constant.EXPORT_CHROME_TRACE: [TraceViewParser]
    }
