# Copyright (c) 2026, Huawei Technologies Co., Ltd

import torch

from torch_npu._compat._impl import compat_impl, compat_impl_container

__all__ = [
    "CustomClassBase",
    "cudagraph_mark_warmup_incomplete",
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


# torch >= 2.15 requires these two fields on WrappedFunction; 2.14 and older
# do not have them at all. PTA does not consume either value, so fill the
# upstream defaults from torch/_inductor/config.py.
def _upstream_managed_input_rerecord_kwargs():
    import dataclasses

    from torch._inductor.cudagraph_utils import WrappedFunction

    # Upstream defaults from torch/_inductor/config.py, used when the field
    # itself has no default. PTA does not consume either value yet.
    defaults = {
        "cudagraph_managed_input_rerecord_limit": 5,
        "cudagraph_managed_input_rerecord_action": "copy",
    }
    fields = {field.name: field for field in dataclasses.fields(WrappedFunction)}
    kwargs = {}
    for name, default in defaults.items():
        field = fields.get(name)
        # Only init=True fields are constructor arguments; dataclasses.fields()
        # also reports init=False fields.
        if field is None or not field.init:
            continue
        kwargs[name] = (
            field.default if field.default is not dataclasses.MISSING else default
        )
    return kwargs


@compat_impl(key="managed_input_rerecord_kwargs", ge=(2, 15))
def managed_input_rerecord_kwargs_upstream():
    return _upstream_managed_input_rerecord_kwargs()


@compat_impl(key="managed_input_rerecord_kwargs", lt=(2, 15))
def managed_input_rerecord_kwargs_local():
    return {}


managed_input_rerecord_kwargs = compat_impl_container[
    "managed_input_rerecord_kwargs"
].apply()


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
        **managed_input_rerecord_kwargs,
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


# Upstream 2.14 added torch.compiler.cudagraph_mark_warmup_incomplete(), which
# resolves torch._inductor.cudagraph_trees.mark_warmup_incomplete() at call
# time. 2.13 has neither, and PTA does not backfill upstream APIs, so the
# capability resolves to None there and callers skip it.
@compat_impl(key="cudagraph_mark_warmup_incomplete", ge=(2, 14))
def cudagraph_mark_warmup_incomplete_upstream():
    return getattr(torch.compiler, "cudagraph_mark_warmup_incomplete", None)


@compat_impl(key="cudagraph_mark_warmup_incomplete", lt=(2, 14))
def cudagraph_mark_warmup_incomplete_local():
    return None


cudagraph_mark_warmup_incomplete = compat_impl_container[
    "cudagraph_mark_warmup_incomplete"
].apply()
