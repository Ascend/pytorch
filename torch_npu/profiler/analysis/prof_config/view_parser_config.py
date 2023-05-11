from ..prof_common_func.constant import Constant
from ..prof_view.kernel_view_parser import KernelViewParser
from ..prof_view.trace_view_parser import TraceViewParser


class ViewParserConfig:
    CONFIG_DICT = {
        Constant.TENSORBOARD_TRACE_HABDLER: [TraceViewParser, KernelViewParser],
        Constant.EXPORT_CHROME_TRACE: [TraceViewParser]
    }
