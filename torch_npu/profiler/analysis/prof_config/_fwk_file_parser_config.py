from ..prof_common_func._file_tag import FileTag
from ..prof_bean._memory_use_bean import MemoryUseBean
from ..prof_bean._op_mark_bean import OpMarkBean
from ..prof_bean._torch_op_bean import TorchOpBean
from ..prof_bean._gc_record_bean import GCRecordBean
from ..prof_bean._python_tracer_hash_bean import PythonTracerHashBean
from ..prof_bean._python_tracer_func_bean import PythonTracerFuncBean
from ..prof_bean._param_tensor_bean import ParamTensorBean


__all__ = []


class FwkFileParserConfig:
    FILE_DISPATCH_MAP = {
        FileTag.TORCH_OP: r"^torch\.op_range",
        FileTag.OP_MARK: r"^torch\.op_mark",
        FileTag.MEMORY: r"^torch\.memory_usage",
        FileTag.GC_RECORD: r"torch\.gc_record",
        FileTag.PYTHON_TRACER_FUNC: r"torch\.python_tracer_func",
        FileTag.PYTHON_TRACER_HASH: r"torch\.python_tracer_hash",
        FileTag.PARAM_TENSOR_INFO: r"torch\.param_tensor_info",
    }

    FILE_BEAN_MAP = {
        FileTag.TORCH_OP: {"bean": TorchOpBean, "is_tlv": True, "struct_size": 58},
        FileTag.OP_MARK: {"bean": OpMarkBean, "is_tlv": True, "struct_size": 40},
        FileTag.MEMORY: {"bean": MemoryUseBean, "is_tlv": True, "struct_size": 76},
        FileTag.GC_RECORD: {"bean": GCRecordBean, "is_tlv": False, "struct_size": 24},
        FileTag.PYTHON_TRACER_FUNC: {"bean": PythonTracerFuncBean, "is_tlv": False, "struct_size": 33},
        FileTag.PYTHON_TRACER_HASH: {"bean": PythonTracerHashBean, "is_tlv": True, "struct_size": 8},
        FileTag.PARAM_TENSOR_INFO: {"bean": ParamTensorBean, "is_tlv": True, "struct_size": 8},
    }
