import dataclasses
import sys
from typing import Any, Callable, overload, TypeVar

import torch


T = TypeVar("T")


def _reduce_without_cached_hash(self):
    fields = dataclasses.fields(self)
    field_values = tuple(getattr(self, field.name) for field in fields if field.init)
    return (self.__class__, field_values)


def _patch_existing_source_classes():
    source_class = torch._guards.Source
    pending = [source_class]
    while pending:
        cls = pending.pop()
        reduce_method = cls.__dict__.get("__reduce__")
        if (
            reduce_method is not None
            and reduce_method.__module__ == "torch._guards"
        ):
            cls.__reduce__ = _reduce_without_cached_hash
        pending.extend(cls.__subclasses__())

    # torch._dynamo.source binds the decorator during import. Keep that binding
    # consistent when the module was imported before torch_npu.
    source_module = sys.modules.get("torch._dynamo.source")
    if source_module is not None:
        source_module.dataclass_with_cached_hash = (
            torch._guards.dataclass_with_cached_hash
        )


def patch_dataclass_with_cached_hash():
    @overload
    def dataclass_with_cached_hash(cls: type[T], **kwargs: Any) -> type[T]: ...

    @overload
    def dataclass_with_cached_hash(
        cls: None = None, **kwargs: Any
    ) -> Callable[[type[T]], type[T]]: ...

    def dataclass_with_cached_hash(
        cls: type[T] | None = None, **kwargs: Any
    ) -> type[T] | Callable[[type[T]], type[T]]:
        def wrap(cls_inner: type[T]) -> type[T]:
            new_cls = dataclasses.dataclass(cls_inner, **kwargs)
            old_hash = cls_inner.__hash__

            def __hash__(self) -> int:
                if not hasattr(self, "_hash"):
                    object.__setattr__(self, "_hash", old_hash(self))
                return self._hash

            new_cls.__hash__ = __hash__
            new_cls.__reduce__ = _reduce_without_cached_hash
            return new_cls  # type: ignore[return-value]

        if cls is None:
            return wrap

        return wrap(cls)

    torch._guards.dataclass_with_cached_hash = dataclass_with_cached_hash
    _patch_existing_source_classes()
