from typing import List

from .gemm_operation import GemmOperation


class Manifest:
    def __init__(self, args=None):
        self.operations = {}
        self.args = args
        self.operation_count = 0

    def append(self, operation, shape_desc) -> None:
        """
        Inserts the operation.

        shape_desc -> {}
            procedural_name -> GemmOperation
        """
        self.operations.setdefault(shape_desc, {}).setdefault(
            operation.procedural_name(), operation
        )
        self.operation_count += 1

    def get_ops(self, shape_desc) -> List[GemmOperation]:
        if shape_desc in self.operations:
            return list(self.operations[shape_desc].values())
        else:
            return None


manifest = Manifest()
