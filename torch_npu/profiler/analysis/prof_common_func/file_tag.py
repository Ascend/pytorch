from enum import Enum


class FileTag(Enum):
    # pytorch file tag
    TORCH_OP = 1
    OP_MARK = 2
    MEMORY = 3
