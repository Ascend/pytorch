# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Single source of truth for the triton-ascend API-generation switch.

IS_TRITON_36_PLUS is True iff the installed triton-ascend exposes the >= 3.6
API surface (vendored triton core >= 3.5.0): launch hooks moved to
triton.knobs.runtime.* AND AttrsDescriptor removed everywhere (constants
passed as a plain dict). False covers triton-ascend 3.2.x (vendored core
3.2.0): hooks are CompiledKernel class attrs, AttrsDescriptor exists.

Capability probe (no version-string parsing), mirroring torch's own
get_triton_attrs_descriptor_version heuristic:
  3.2.2 -> (no knobs, has AttrsDescriptor) -> False
  3.6   -> (has knobs, no AttrsDescriptor) -> True
A hypothetical mixed shape (knobs present, AttrsDescriptor still there)
resolves to False -- the safe/legacy direction. Equivalent to torch's
triton_version_uses_attrs_dict() on every released triton-ascend.
"""

try:
    from triton import knobs  # noqa: F401

    _has_knobs = True
except ImportError:
    _has_knobs = False

_has_attrs_descriptor = False
try:
    import triton.backends.compiler as _backends_compiler

    _has_attrs_descriptor |= hasattr(_backends_compiler, "AttrsDescriptor")
except ImportError:
    pass
try:
    import triton.compiler.compiler as _compiler_compiler

    _has_attrs_descriptor |= hasattr(_compiler_compiler, "AttrsDescriptor")
except ImportError:
    pass

IS_TRITON_36_PLUS = _has_knobs and not _has_attrs_descriptor
