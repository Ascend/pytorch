from enum import Enum

__all__ = []


class FileTag(Enum):
    # pytorch file tag
    TORCH_OP = 1
    OP_MARK = 2
    MEMORY = 3
    GC_RECORD = 6
    PYTHON_TRACER_FUNC = 7
    PYTHON_TRACER_HASH = 8
    PARAM_TENSOR_INFO = 9
