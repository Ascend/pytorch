from ..prof_common_func.file_tag import FileTag
from ..prof_bean.memory_use_bean import MemoryUseBean
from ..prof_bean.op_mark_bean import OpMarkBean
from ..prof_bean.torch_op_bean import TorchOpBean
from ..prof_bean.python_module_call_bean import PythonModuleCallBean
from ..prof_bean.python_func_call_bean import PythonFuncCallBean

__all__ = []


class FwkFileParserConfig:
    FILE_DISPATCH_MAP = {
        FileTag.TORCH_OP: r"^torch\.op_range",
        FileTag.OP_MARK: r"^torch\.op_mark",
        FileTag.MEMORY: r"^torch\.memory_usage",
        FileTag.PYTHON_FUNC_CALL: r"^torch\.python_func_call",
        FileTag.PYTHON_MODULE_CALL: r"^torch\.python_module_call"
    }

    FILE_BEAN_MAP = {
        FileTag.TORCH_OP: {"bean": TorchOpBean, "is_tlv": True, "struct_size": 57},
        FileTag.OP_MARK: {"bean": OpMarkBean, "is_tlv": True, "struct_size": 40},
        FileTag.MEMORY: {"bean": MemoryUseBean, "is_tlv": True, "struct_size": 76},
        FileTag.PYTHON_FUNC_CALL: {"bean": PythonFuncCallBean, "is_tlv": True, "struct_size": 25},
        FileTag.PYTHON_MODULE_CALL: {"bean": PythonModuleCallBean, "is_tlv": True, "struct_size": 24}
    }
