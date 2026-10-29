# Copyright (c) 2026, Huawei Technologies Co., Ltd

from torch_npu._compat._impl import compat_impl, compat_impl_container

__all__ = [
    "CustomClassBase",
    "is_opaque_input",
    "make_wrapped_function",
    "register_opaque_input",
]


# torch >= 2.14 renamed OpaqueBase/is_opaque_value to
# CustomClassBase/is_custom_class_obj.
@compat_impl(key="opaque_base_type", ge=(2, 14))
def opaque_base_type_upstream():
    from torch._custom_class_base import CustomClassBase

    return CustomClassBase


@compat_impl(key="opaque_base_type", lt=(2, 14))
def opaque_base_type_local():
    from torch._opaque_base import OpaqueBase

    return OpaqueBase


opaque_base_type = compat_impl_container["opaque_base_type"].apply()
CustomClassBase = opaque_base_type


@compat_impl(key="is_opaque_input", ge=(2, 14))
def is_opaque_input_upstream(value):
    from torch._library.opaque_object import is_custom_class_obj

    return is_custom_class_obj(value)


@compat_impl(key="is_opaque_input", lt=(2, 14))
def is_opaque_input_local(value):
    from torch._library.opaque_object import is_opaque_value

    return is_opaque_value(value)


is_opaque_input = compat_impl_container["is_opaque_input"].resolve()


@compat_impl(key="register_opaque_input", ge=(2, 14))
def register_opaque_input_upstream(cls):
    from torch._library.opaque_object import register_custom_class

    register_custom_class(cls, typ="symbolic")


@compat_impl(key="register_opaque_input", lt=(2, 14))
def register_opaque_input_local(cls):
    from torch._library.opaque_object import register_opaque_type

    register_opaque_type(cls, typ="reference")


register_opaque_input = compat_impl_container["register_opaque_input"].resolve()


@compat_impl(key="make_wrapped_function", ge=(2, 14))
def make_wrapped_function_upstream(
    model,
    static_input_idxs,
    function_id,
    constants,
    placeholders,
    mutated_input_idxs,
    *,
    kernel_free_cudagraph=False,
    user_visible_output_idxs=(),
):
    from torch._inductor.cudagraph_utils import WrappedFunction

    return WrappedFunction(
        model,
        static_input_idxs,
        function_id,
        constants,
        placeholders,
        mutated_input_idxs,
        kernel_free_cudagraph=kernel_free_cudagraph,
        user_visible_output_idxs=frozenset(user_visible_output_idxs),
    )


@compat_impl(key="make_wrapped_function", lt=(2, 14))
def make_wrapped_function_local(
    model,
    static_input_idxs,
    function_id,
    constants,
    placeholders,
    mutated_input_idxs,
    *,
    kernel_free_cudagraph=False,
    user_visible_output_idxs=(),
):
    import dataclasses

    from torch._inductor.cudagraph_utils import WrappedFunction

    # torch 2.13's WrappedFunction has neither of these fields. Add them on a
    # local subclass so the graph-tree code can use the same attributes on all
    # supported PyTorch versions.
    @dataclasses.dataclass(frozen=True)
    class _LocalWrappedFunction(WrappedFunction):
        kernel_free_cudagraph: bool = False
        user_visible_output_idxs: frozenset = dataclasses.field(
            default_factory=frozenset
        )

    return _LocalWrappedFunction(
        model,
        static_input_idxs,
        function_id,
        constants,
        placeholders,
        mutated_input_idxs,
        kernel_free_cudagraph=kernel_free_cudagraph,
        user_visible_output_idxs=frozenset(user_visible_output_idxs),
    )


make_wrapped_function = compat_impl_container["make_wrapped_function"].resolve()
