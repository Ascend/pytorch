# Copyright (c) 2026, Huawei Technologies Co., Ltd

from torch_npu._compat._impl import compat_impl, compat_impl_container

__all__ = ["CustomClassBase", "is_opaque_input", "register_opaque_input"]


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
