# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from __future__ import annotations

import os
from typing import Any

from torch._inductor.runtime.triton_helpers import get_constexprs

from ..compat import IS_TRITON_36_PLUS
from .kernel import NPUStaticallyLaunchedTritonKernel
from .metadata import (
    NPUHostArgsLayout,
    NPULaunchMetadata,
    NPU_STATIC_LAUNCH_SCHEMA_VERSION,
)


_TYPE_TO_ARG_KIND = {
    "i1": "bool",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "u1": "u32",
    "u8": "u8",
    "u16": "u16",
    "u32": "u32",
    "u64": "u64",
    "fp16": "f32",
    "bf16": "f32",
    "fp32": "f32",
    "f32": "f32",
    "fp64": "f64",
}


class NPUStaticArtifactError(RuntimeError):
    pass


_MISSING = object()


def _metadata_value(metadata: Any, name: str, default: Any) -> Any:
    if isinstance(metadata, dict):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


def _required_metadata_value(metadata: Any, name: str) -> Any:
    value = _metadata_value(metadata, name, _MISSING)
    if value is _MISSING or value is None:
        raise NPUStaticArtifactError(
            f"Compiled kernel metadata field {name!r} is missing"
        )
    return value


def _signature_index(arg_names: tuple[str, ...], key: Any) -> int:
    if isinstance(key, str):
        try:
            return arg_names.index(key)
        except ValueError as exc:
            raise NPUStaticArtifactError(f"Unknown signature argument: {key}") from exc
    if isinstance(key, tuple):
        if not key:
            raise NPUStaticArtifactError("Empty tuple is not a signature key")
        key = key[0]
    if not isinstance(key, int):
        raise NPUStaticArtifactError(f"Unsupported signature key: {key!r}")
    if key < 0 or key >= len(arg_names):
        raise NPUStaticArtifactError(f"Signature index is out of range: {key}")
    return key


def _arg_kind(signature: Any) -> str:
    text = str(signature)
    if text.startswith("*"):
        return "tensor"
    if text.startswith("tensordesc"):
        raise NPUStaticArtifactError("Tensor descriptor arguments are unsupported")
    try:
        return _TYPE_TO_ARG_KIND[text]
    except KeyError as exc:
        raise NPUStaticArtifactError(
            f"Unsupported NPU static launcher argument type: {text}"
        ) from exc


def _npubin_path(binary: Any) -> str | None:
    metadata_group = getattr(binary, "metadata_group", {}) or {}
    for filename, path in metadata_group.items():
        if str(filename).endswith(".npubin"):
            return str(path)
    return None


def _target_supports_ffts(arch: str) -> bool:
    try:
        from triton.backends.ascend.utils import (
            force_disable_ffts,
            is_ffts_supported,
        )

        try:
            disabled = force_disable_ffts(arch)
        except TypeError:
            # Compatibility with Triton-Ascend releases where the helper read
            # the active target internally and accepted no explicit arch.
            disabled = force_disable_ffts()
        return bool(is_ffts_supported(arch) and not disabled)
    except (ImportError, TypeError, ValueError) as exc:
        raise NPUStaticArtifactError(
            f"Unable to determine the FFTS ABI for target {arch!r}"
        ) from exc


class NPUStaticArtifactAdapter:
    @staticmethod
    def from_compiled_kernel(
        binary: Any,
        compile_meta: dict[str, Any],
        inductor_meta: dict[str, Any],
    ) -> NPUStaticallyLaunchedTritonKernel:
        asm = getattr(binary, "asm", {}) or {}
        npubin_raw = asm.get("npubin")
        if not isinstance(npubin_raw, bytes) or not npubin_raw:
            raise NPUStaticArtifactError("Compiled kernel has no npubin binary")

        src = getattr(binary, "src", None)
        fn = getattr(src, "fn", None)
        raw_arg_names = getattr(fn, "arg_names", None)
        if raw_arg_names is None:
            raise NPUStaticArtifactError("Compiled kernel argument names are missing")
        arg_names = tuple(raw_arg_names)

        signature = getattr(src, "signature", None)
        if not isinstance(signature, dict):
            raise NPUStaticArtifactError("Compiled kernel signature is missing")
        constants = getattr(src, "constants", {}) or {}
        constant_indices = {
            _signature_index(arg_names, key) for key in constants.keys()
        }
        indexed_signature = {
            _signature_index(arg_names, key): value
            for key, value in signature.items()
        }
        runtime_arg_indices = tuple(
            index
            for index in sorted(indexed_signature)
            if indexed_signature[index] != "constexpr" and index not in constant_indices
        )
        arg_kinds = tuple(
            _arg_kind(indexed_signature[index]) for index in runtime_arg_indices
        )

        declared_constexprs = tuple(get_constexprs(fn))
        if IS_TRITON_36_PLUS:
            implicit_constants = {
                arg_names[index] for index in declared_constexprs
            } | {"num_warps", "num_stages"}
            implicit_constants &= set(compile_meta.get("constants", {}).keys())
            launcher_arg_names = tuple(
                name for name in arg_names if name not in implicit_constants
            )
        else:
            known_constants = {
                arg_names[index] for index in declared_constexprs
            }
            none_args = {
                name
                for name, value in compile_meta.get("constants", {}).items()
                if value is None and name not in known_constants
            }
            none_args -= set(compile_meta.get("signature", {}).keys())
            launcher_arg_names = tuple(
                name
                for index, name in enumerate(arg_names)
                if index not in declared_constexprs and name not in none_args
            )

        metadata = getattr(binary, "metadata", None)
        target = _metadata_value(metadata, "target", None)
        target_arch = str(getattr(target, "arch", "") or "")
        if not target_arch:
            raise NPUStaticArtifactError("Compiled kernel target arch is missing")

        # Keep the metadata-name translation beside the host ABI selection.
        # Sources:
        #   3.2.2: third_party/ascend/backend/compiler.py::NPUOptions and
        #          third_party/ascend/backend/driver.py::generate_npu_wrapper_src
        #   3.6.0: third_party/ascend/backend/compiler.py::NPUOptions and
        #          third_party/ascend/backend/driver.py::make_launcher
        # C++ receives only normalized values; it must not inspect the Triton
        # version or these version-specific metadata names.
        if IS_TRITON_36_PLUS:
            host_args_layout = NPUHostArgsLayout.TRITON_ASCEND_3_6
            pure_simt_field = "is_pure_simt"
            lock_num = _metadata_value(metadata, "sync_block_lock_layout", 0)
        else:
            host_args_layout = NPUHostArgsLayout.TRITON_ASCEND_3_2
            pure_simt_field = "force_simt_only"
            lock_num = _metadata_value(metadata, "lock_num", 0)

        mix_mode = str(_required_metadata_value(metadata, "mix_mode")).lower()
        parallel_mode = str(
            _required_metadata_value(metadata, "parallel_mode")
        ).lower()
        is_pure_simt = bool(_required_metadata_value(metadata, pure_simt_field))
        # Zero-sized scratch still has pointer slots in the 3.6 host ABI, but
        # this launcher does not allocate storage for an active requirement.
        for scratch_size_field in (
            "global_scratch_size",
            "profile_scratch_size",
        ):
            scratch_size = int(_metadata_value(metadata, scratch_size_field, 0) or 0)
            if scratch_size > 0:
                raise NPUStaticArtifactError(
                    f"{scratch_size_field} is required but unsupported"
                )
        launch_metadata = NPULaunchMetadata(
            schema_version=NPU_STATIC_LAUNCH_SCHEMA_VERSION,
            host_args_layout=host_args_layout,
            target_arch=target_arch,
            mix_mode=mix_mode,
            parallel_mode=parallel_mode,
            enable_simt="simt" in parallel_mode or is_pure_simt,
            shared_mem_dynamic_size=int(
                _required_metadata_value(metadata, "shared_mem_dynamic_size")
            ),
            is_pure_simt=is_pure_simt,
            target_support_ffts=_target_supports_ffts(target_arch),
            num_ctas=int(_metadata_value(metadata, "num_ctas", 1) or 1),
            cluster_dims=tuple(
                _metadata_value(metadata, "cluster_dims", (1, 1, 1))
                or (1, 1, 1)
            ),
            workspace_size=int(_metadata_value(metadata, "workspace_size", 0) or 0),
            lock_num=int(lock_num or 0),
            device_print_enabled=os.getenv(
                "TRITON_DEVICE_PRINT", "false"
            ).lower()
            in ("true", "1"),
        )
        launch_metadata.validate()

        return NPUStaticallyLaunchedTritonKernel(
            name=str(getattr(binary, "name", "") or _metadata_value(metadata, "name", "")),
            npubin_raw=npubin_raw,
            npubin_path=_npubin_path(binary),
            kernel_hash=str(getattr(binary, "hash", "")),
            arg_names=arg_names,
            launcher_arg_names=launcher_arg_names,
            runtime_arg_names=tuple(arg_names[index] for index in runtime_arg_indices),
            declared_constexprs=declared_constexprs,
            full_constexprs=tuple(sorted(constant_indices)),
            arg_kinds=arg_kinds,
            launch_metadata=launch_metadata,
            device=int(compile_meta.get("device", 0) or 0),
            num_warps=int(_metadata_value(metadata, "num_warps", 1) or 1),
            shared=int(
                getattr(binary, "shared", _metadata_value(metadata, "shared", 0)) or 0
            ),
            n_regs=getattr(binary, "n_regs", None),
            n_spills=getattr(binary, "n_spills", None),
        )


__all__ = ["NPUStaticArtifactAdapter", "NPUStaticArtifactError"]
