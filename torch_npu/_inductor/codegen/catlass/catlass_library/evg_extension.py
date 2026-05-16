"""
Base class for Python EVG fronted
"""

from __future__ import annotations

from typing import Callable, Dict, Union

from sympy import Expr
from catlass_cppgen.common.op_tensor import OpTensor
from catlass_cppgen.common.data_type import DataType


def create_example_tensors(
    var_name_to_buffer_name: Dict[str, str],
    name_to_buffer: Dict[str, Buffer],
    size_hint_fn: Callable[[Union[Expr, int]], int],
) -> Dict[str, OpTensor]:
    def catlass_tensor_from_buffer(buffer: Buffer) -> OpTensor:
        shape = tuple(buffer.get_layout().size)
        stride = tuple(buffer.get_layout().stride)
        return OpTensor.from_shape_stride(
            shape=shape,
            stride=stride,
            dtype=DataType.from_dtype(buffer.get_layout().dtype),
        )

    return {
        key: catlass_tensor_from_buffer(name_to_buffer[name])
        for key, name in var_name_to_buffer_name.items()
    }
