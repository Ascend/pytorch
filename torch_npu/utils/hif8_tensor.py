# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with HIF8 data"""
from __future__ import annotations

__all__ = ["HiFloat8Tensor"]

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils._pytree import tree_map
import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error


# init transformer engine
torch_npu._C._cd_init()

tex = torch_npu._C._cd
aten = torch.ops.aten

NPU_CUSTOM_DType = {
    torch.uint8: tex.DType.uint8,
    torch.int32: tex.DType.int32,
    torch.float32: tex.DType.float32,
    torch.half: tex.DType.float16,
    torch.bfloat16: tex.DType.bfloat16,
}


class _FromHiFloat8Func(torch.autograd.Function):
    """Cast from HIF8 to other dtype"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: HiFloat8Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = tensor.dtype
        data = tensor._data.contiguous().view(1, -1).detach()
        out = tex.cast_from_fp8(
            data,
            tex.DType.hifloat8,
            NPU_CUSTOM_DType[dtype],
        )
        out = out.view(tensor.size())
        return out

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # Assume that we want gradients in full precision
        return grad, None


class _ToHiFloat8Func(torch.autograd.Function):
    """Cast to HIF8 from other dtype"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: torch.Tensor,
    ) -> HiFloat8Tensor:

        # Check input tensor TODO
        tensor = tensor.contiguous().npu().detach()
        if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            tensor = tensor.float()

        # Cast data to HIF8
        data = tex.cast_to_fp8(
            tensor.view(1, -1),
            tex.DType.hifloat8,
        )
        data = data.view(tensor.size())

        # Construct HIF8 tensor
        return HiFloat8Tensor(
            data=data,
            dtype=tensor.dtype,
        )

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # Assume that we want gradients in full precision
        return grad, None


class _IdentityFunc(torch.autograd.Function):
    """Identity function

    If constructor keyword-arguments are provided, then construct a
    new HiFloat8Tensor using the provided tensor's attributes.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: HiFloat8Tensor,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:

        # Return input tensor if constructor kwargs are not provided
        ctx.input_dtype = tensor.dtype
        if init_kwargs is None:
            return tensor

        # Construct new tensor if constructor kwargs are provided
        default_kwargs = dict(
            data=tensor._data,
            dtype=tensor.dtype,
        )
        for key, val in default_kwargs.items():
            if key not in init_kwargs:
                init_kwargs[key] = val
        return HiFloat8Tensor(**init_kwargs)

    @staticmethod
    def backward(ctx, grad):
        return grad.to(ctx.input_dtype), None


class _ViewFunc(torch.autograd.Function):
    """View function

    View the HiFloat8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        shape: Tuple[int] = None,
    ) -> torch.Tensor:

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Construct new tensor if shape is provided
        if isinstance(tensor, HiFloat8Tensor):
            return HiFloat8Tensor.make_like(
                tensor,
                data=tensor._data.view(*shape),
            )
        return tensor.view(*shape)

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        if isinstance(grad, HiFloat8Tensor):
            dgrad = HiFloat8Tensor.make_like(
                grad,
                data=grad._data.view(ctx.shape),
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the HiFloat8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        shape: Tuple[int] = None,
    ) -> torch.Tensor:

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Construct new tensor if shape is provided
        if isinstance(tensor, HiFloat8Tensor):
            return HiFloat8Tensor.make_like(
                tensor,
                data=tensor._data.reshape(*shape),
            )
        return tensor.reshape(*shape)

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        if isinstance(grad, HiFloat8Tensor):
            dgrad = HiFloat8Tensor.make_like(
                grad,
                data=grad._data.reshape(ctx.shape),
            )
            return dgrad, None
        return grad.reshape(ctx.shape), None


class _TransposeFunc(torch.autograd.Function):
    """Transpose function

    Transpose the HiFloat8Tensor.

    """

    @staticmethod
    def forward(ctx, tensor, dim0, dim1):
        ctx.save_for_backward(dim0, dim1)
        if isinstance(tensor, HiFloat8Tensor):
            return HiFloat8Tensor.make_like(
                tensor,
                data=tensor._data.transpose(dim0, dim1),
            )
        return tensor.transpose(dim0, dim1)

    @staticmethod
    def backward(ctx, grad):
        dim0, dim1 = ctx.saved_tensors
        if isinstance(grad, HiFloat8Tensor):
            dgrad = HiFloat8Tensor.make_like(
                grad,
                data=grad._data.transpose(dim0, dim1),
            )
            return dgrad, None
        return grad.transpose(dim0, dim1), None, None


class HiFloat8Tensor(torch.Tensor):
    """Experimental tensor class with HIF8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) HIF8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    data: torch.Tensor
          Raw HIF8 data in a uint8 tensor
    dtype: torch.dtype, default = torch.float32
           Nominal tensor datatype.

    """

    def __new__(
        cls,
        *,
        data: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ):
        # Check that data buffer is valid
        if data.element_size() != 1:
            raise ValueError(
                f"HiFloat8Tensor requires data buffer with 8-bit dtype (got dtype={data.dtype})"
                + pta_error(ErrCode.VALUE)
            )
        if data.requires_grad:
            raise ValueError(
                "HiFloat8Tensor requires non-differentiable data buffer"
                + pta_error(ErrCode.VALUE)
            )
        if not data.is_npu:
            data = data.npu()

        # Initialize tensor object
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data: torch.Tensor = data

        return self

    @classmethod
    def make_like(
        cls,
        tensor: HiFloat8Tensor,
        *,
        data: torch.Tensor,
        **kwargs,
    ) -> HiFloat8Tensor:
        """Use attributes of a HiFloat8Tensor to create another HiFloat8Tensor

        See constructor for list of keyword arguments.

        """
        default_kwargs = dict(
            dtype=tensor.dtype,
        )
        for key, val in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = val
        return HiFloat8Tensor(data=data, **kwargs)

    def __repr__(self):
        return (
            "HiFloat8Tensor("
            f"data={self.from_hifloat8(dtype=self.dtype)}"
            ")"
        )

    def from_hifloat8(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct PyTorch tensor from HiFloat8Tensor

        By default the resulting tensor's dtype is the
        HiFloat8Tensor's nominal dtype.
        """
        return _FromHiFloat8Func.apply(self, dtype)

    @classmethod
    def to_hifloat8(
        cls,
        tensor: torch.Tensor
    ):
        """Construct HiFloat8Tensor from PyTorch tensor"""
        return _ToHiFloat8Func.apply(
            tensor
        )

    def float(self) -> torch.Tensor:
        return self.from_hifloat8(dtype=torch.float32)

    def bfloat16(self) -> torch.Tensor:
        return self.from_hifloat8(dtype=torch.bfloat16)

    def half(self) -> torch.Tensor:
        return self.from_hifloat8(dtype=torch.float16)

    def cpu(self) -> torch.Tensor:
        return self.from_hifloat8().cpu()

    def clone(self) -> HiFloat8Tensor:
        return _IdentityFunc.apply(self, {"data": self._data.detach().clone()})

    def view(self, *shape: Tuple[int]) -> HiFloat8Tensor:
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> HiFloat8Tensor:
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        *,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> HiFloat8Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._data.is_contiguous(memory_format=memory_format):
            return self
        return _IdentityFunc.apply(
            self,
            {"data": self._data.detach().contiguous(memory_format=memory_format)},
        )

    def to_dtype(self, dtype: torch.dtype) -> HiFloat8Tensor:
        """Create `HiFloat8Tensor` with given nominal dtype

        The new tensor has the same underlying HIF8 data.

        """
        return HiFloat8Tensor.make_like(
            self,
            data=self._data,
            dtype=dtype,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # In-place copy op
        if func == aten.copy_.default:

            # Check tensors
            dst = args[0]
            src = args[1]
            if not isinstance(dst, torch.Tensor):
                raise RuntimeError(
                    "Attempted to copy into something that isn't a PyTorch tensor"
                    + pta_error(ErrCode.TYPE)
                )
            if not isinstance(src, torch.Tensor):
                raise RuntimeError(
                    "Attempted to copy from something that isn't a PyTorch tensor"
                    + pta_error(ErrCode.TYPE)
                )

            # Special handling based on which tensors are HIF8
            dst_is_hif8 = isinstance(dst, HiFloat8Tensor)
            src_is_hif8 = isinstance(src, HiFloat8Tensor)
            if dst_is_hif8 and src_is_hif8:
                # Directly copy HIF8 data if possible
                dst._data.copy_(src._data)

            elif not dst_is_hif8 and src_is_hif8:
                # Cast source tensor to higher precision
                dst.copy_(src.from_hifloat8())

            elif dst_is_hif8 and not src_is_hif8:
                # Make sure input is in expected format
                src = src.expand(dst.size())
                src = src.to(
                    device=dst.device,
                    memory_format=torch.contiguous_format,
                )

                # Cast to HIF8
                if not dst._data.is_contiguous():
                    raise RuntimeError(
                        "Transformer Engine cast kernels require contiguous data"
                        + pta_error(ErrCode.INTERNAL)
                    )
                tex.cast_to_fp8_noalloc(
                    src.view(1, -1),
                    dst._data.view(1, -1),
                    tex.DType.hifloat8,
                )
            else:
                # Invalid case
                raise RuntimeError(
                    "Using HiFloat8Tensor copy logic, but no HiFloat8Tensor found"
                    + pta_error(ErrCode.INTERNAL)
                )

            # Nothing to return for in-place ops
            return None

        # Slice op
        if func == aten.slice.Tensor:
            tensor = args[0]
            data = tensor._data
            data_slice = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return HiFloat8Tensor.make_like(tensor, data=data_slice)

        # Detach op
        if func == aten.detach.default:
            # Simply return a new HiFloat8Tensor with the same attrs
            return HiFloat8Tensor.make_like(
                args[0],
                data=args[0]._data,
            )

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._data
            data_view = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return HiFloat8Tensor.make_like(
                tensor,
                data=data_view,
            )

        def maybe_unwrap(t):
            if isinstance(t, HiFloat8Tensor):
                return t.from_hifloat8()
            return t

        def maybe_update_inplace(arg, new_arg, schema_arg):
            """Update values of HIF8 tensors

            Keep the same HIF8 scaling factors.

            """
            check_args = isinstance(arg, HiFloat8Tensor) and isinstance(new_arg, torch.Tensor)
            check_schema = (
                hasattr(schema_arg, "alias_info")
                and hasattr(schema_arg.alias_info, "is_write")
                and schema_arg.alias_info.is_write
            )

            if check_args and check_schema:
                arg.copy_(new_arg)

        # In-place op
        if func._schema.is_mutable:
            # Cast to higher precision, perform op, and cast values
            # back to original HIF8 buffers
            new_args = tree_map(maybe_unwrap, args)
            new_kwargs = tree_map(maybe_unwrap, kwargs)
            schema_args = func._schema.arguments
            args_len = len(args)
            out = super().__torch_dispatch__(func, types, new_args, new_kwargs)
            for arg, new_arg, schema_arg in zip(args, new_args, schema_args):
                maybe_update_inplace(arg, new_arg, schema_arg)
            for kwarg, new_kwarg, schema_arg in zip(kwargs, new_kwargs, schema_args[args_len:]):
                if not (kwarg == new_kwarg == schema_arg.name):
                    raise ValueError('name of the kw argument should match' + pta_error(ErrCode.VALUE))
                maybe_update_inplace(kwargs[kwarg], new_kwargs[new_kwarg], schema_arg)
            return None

        # Default op
        # Note: cast to higher precision and perform op
        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    @classmethod
    def _make_in_reduce_ex(
        cls,
        data: torch.Tensor,
        dtype: torch.dtype,
    ) -> HiFloat8Tensor:
        """Build HiFloat8Tensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return HiFloat8Tensor(
            data=data,
            dtype=dtype,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling to remove references to HIF8 metadata objects"""
        return (
            HiFloat8Tensor._make_in_reduce_ex,
            (self._data, self.dtype),
        )

    def _get_data(self) -> HiFloat8Tensor:
        """Get tensor data property"""
        return super().data

    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Cast tensor to HIF8 and store in HIF8 buffer.

        """
        with torch.no_grad():
            self.copy_(tensor)

    # Cast to HIF8 when setting HiFloat8Tensor.data
    data = property(_get_data, _set_data)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs)

    def transpose(self, dim0, dim1):
        return _TransposeFunc.apply(self, dim0, dim1)
