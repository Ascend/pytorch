from enum import Enum

__all__ = []


class FileTag(Enum):
    # pytorch file tag
    TORCH_OP = 1
    OP_MARK = 2
    MEMORY = 3
    PYTHON_FUNC_CALL = 4
    PYTHON_MODULE_CALL = 5
