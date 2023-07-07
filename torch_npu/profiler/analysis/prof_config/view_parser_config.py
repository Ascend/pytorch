from ..prof_common_func.constant import Constant
from ..prof_view.kernel_view_parser import KernelViewParser
from ..prof_view.operator_view_parser import OperatorViewParser
from ..prof_view.trace_view_parser import TraceViewParser
from ..prof_view.memory_view_parser import MemoryViewParser
from ..prof_view.integrate_parser import IntegrateParser


class ViewParserConfig:
    CONFIG_DICT = {
        Constant.TENSORBOARD_TRACE_HABDLER: [OperatorViewParser, TraceViewParser, KernelViewParser,
                                             MemoryViewParser, IntegrateParser],
        Constant.EXPORT_CHROME_TRACE: [TraceViewParser]
    }
