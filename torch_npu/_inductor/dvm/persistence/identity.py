# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""DVM producer identity: the values a compile's cache keys are built from.

Two halves, both DVM-specific:

* the **configuration projection** -- which DVM / ANIR / shape knobs a
  compilation is keyed on, and which are initialization-time only;
* the **installation fingerprint** -- the DVM modules and ``libtorch_npu`` the
  artifact was built with.

The generic format (canonical form, hashing, ``Manifest``) lives in
:mod:`torch_npu.dynamo._package_compat.format` and is imported from there.
"""
import copy
import functools
import importlib
import sys
from pathlib import Path

from torch_npu.dynamo._package_compat.format import (
    SCHEMA, IdentityUnavailable, canonical, stable_hash,
)

# Shared controls required by pure DVM. Mixed graphs temporarily keep this same
# projection; Triton-specific configuration must remain unchanged for reuse.
# In-place policy can affect CPU nodes; graph-tree checks are backend-independent.
# DVM lowering/fallback/accuracy use ANIR_FIELDS, not the similarly named public
# Triton controls. Diagnostic paths and host parallelism do not invalidate code.
COMMON_FIELDS = (
    "enable_inplace_buffers",
    "npugraph_trees.disable_cpu_input_check",
)
# Configure decomposition lists and addmm fusion policy before DVM initialization.
# Callable fusion policies must provide uuid(); canonical() rejects opaque ones.
DVM_FIELDS = (
    "enable_cat", "enable_fusion_permute", "enable_decomp_list", "disable_decomp_list",
    "vf_fusion", "view_fusion_level", "disable_post_reduce_fusion", "enable_matmul_fusion",
    "disable_addmm_fusion", "matmul_fusion_rule", "bf16_vector_keep_promoted",
    # Migrated into dvm.config when the DVM backend was decoupled from MLIR; the
    # DVM path reads these from the new module, so the identity must follow.
    "online_acc_comp", "enable_full_lowering_fallback", "force_fallback_kernel_names",
    "GENERATE_LIST", "decomps_to_exclude_npu",
)
ANIR_FIELDS = (
    "always_compile", "enable_graph_trace", "acc_comp_mode", "disable_any_pbr",
    "autotune_fx_fallback", "cache_named_op",
    "acc_check_during_tune", "fallback_fold_expand", "acc_comp_tol",
    "autotune", "mode", "extra_command",
    "force_fallback_kernel_paths", "compile_mode", "block_dim", "fallback_to_aten_mode",
    "FALLBACK_LIST",
)
INITIALIZATION_FIELDS = {
    "torch_npu._inductor.dvm.config": (
        "enable_decomp_list", "disable_decomp_list", "enable_matmul_fusion", "disable_addmm_fusion",
        "decomps_to_exclude_npu",
    ),
}
_DVM_INITIALIZATION = None


def _getattr_path(module, name):
    for part in name.split("."):
        module = getattr(module, part)
    return module


def _snapshot(module_name, fields):
    module = importlib.import_module(module_name)
    result = {}
    for field in fields:
        try:
            value = _getattr_path(module, field)
        except AttributeError:
            # An absent required field is a format/version mismatch, not a default.
            raise IdentityUnavailable(f"NPU_CONFIG_FIELD_UNAVAILABLE:{module_name}.{field}") from None
        result[field] = _cache_snapshot(value, f"{module_name}.{field}")
    canonical(result)
    return result


def _config_identity(value, label):
    try:
        return canonical(value)
    except IdentityUnavailable:
        raise
    except Exception as error:
        raise IdentityUnavailable(
            f"NPU_CONFIG_IDENTITY_UNAVAILABLE:{label}:{type(error).__name__}:{error}") from error


def _cache_snapshot(value, label):
    """Copy configuration containers while keeping UUID-bearing objects live."""
    _config_identity(value, label)
    memo = {}

    def preserve_callbacks(item):
        if isinstance(item, dict):
            for key, child in item.items():
                preserve_callbacks(key)
                preserve_callbacks(child)
        elif isinstance(item, (tuple, list, set, frozenset)):
            for child in item:
                preserve_callbacks(child)
        elif callable(getattr(item, "uuid", None)):
            memo[id(item)] = item

    try:
        preserve_callbacks(value)
        return copy.deepcopy(value, memo)
    except Exception as error:
        raise IdentityUnavailable(
            f"NPU_CONFIG_SNAPSHOT_UNAVAILABLE:{label}:{type(error).__name__}:{error}") from error


def _options_identity(options):
    return _config_identity({key: value for key, value in options.items()
                             if key not in ("npu_backend", "compile_threads") and not key.startswith("trace.")},
                            "compiler_options")


def record_initialization():
    """Called only after DVM actually installs its initialization-time policies."""
    global _DVM_INITIALIZATION
    try:
        _DVM_INITIALIZATION = canonical({name: _snapshot(name, fields)
                                         for name, fields in INITIALIZATION_FIELDS.items()})
    except Exception as error:
        _DVM_INITIALIZATION = str(error) or type(error).__name__


def initialization_state(configs=None):
    if isinstance(_DVM_INITIALIZATION, str):
        raise IdentityUnavailable(f"NPU_INITIALIZATION_UNAVAILABLE:{_DVM_INITIALIZATION}")
    if not isinstance(_DVM_INITIALIZATION, dict):
        raise IdentityUnavailable("NPU_INITIALIZATION_UNAVAILABLE")
    if configs is None:
        configs = {name: _snapshot(name, fields) for name, fields in INITIALIZATION_FIELDS.items()}
    requested = canonical({name: {field: configs[name][field] for field in fields}
                           for name, fields in INITIALIZATION_FIELDS.items()})
    if requested != _DVM_INITIALIZATION:
        raise IdentityUnavailable("NPU_INITIALIZATION_CONFIG_MISMATCH")
    from torch._inductor import decomposition

    # Built-in implementations belong to the versioned torch/torch_npu code.
    # User replacements need a UUID, just like custom codegen passes.
    tables = {name: stable_hash([(str(op), _decomposition_key(fn))
                                 for op, fn in sorted(getattr(decomposition, name).items(),
                                                      key=lambda item: str(item[0]))])
              for name in ("decompositions", "extra_random_decomps")}
    return {"configs": _DVM_INITIALIZATION, "tables": tables}


def _decomposition_key(fn):
    if isinstance(fn, functools.partial):
        return {"partial": _decomposition_key(fn.func), "args": canonical(fn.args),
                "kwargs": canonical(fn.keywords)}
    module = getattr(fn, "__module__", "") or ""
    name = getattr(fn, "__qualname__", getattr(fn, "__name__", None))
    if name and module.split(".")[0] in ("torch", "torch_npu"):
        return {"module": module, "name": name}
    return canonical(fn)


def capture_configs(backend):
    if backend != "dvm":
        raise IdentityUnavailable("NPU_BACKEND_OUT_OF_SCOPE:" + backend)
    configs = {"torch_npu._inductor.config": _snapshot("torch_npu._inductor.config", COMMON_FIELDS)}
    name = "torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.config"
    configs[name] = _snapshot(name, ANIR_FIELDS)
    name = "torch_npu._inductor.dvm.config"
    configs[name] = _snapshot(name, DVM_FIELDS)
    canonical(configs)
    return configs


@functools.cache
def _digest_file(path):
    # Like upstream torch_key(), compiler installations
    # are immutable for the lifetime of a process. Restart after upgrading them.
    import hashlib

    before = path.stat()
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise OSError("file changed while hashing")
    return digest.hexdigest()


def file_digest(path):
    """Content identity, computed once per resolved file in this process.

    This is not a stat-based content guess. New processes hash their own files;
    changing an installed compiler/library in-place requires restarting processes.
    """
    try:
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise OSError("not a regular file")
        return _digest_file(path)
    except (OSError, TypeError) as error:
        raise IdentityUnavailable(f"NPU_DEPENDENCY_UNREADABLE:{path}:{type(error).__name__}:{error}") from error


def collect_identity(devices, backend):
    """Fingerprint the compiler installation a DVM artifact was built with."""
    if backend != "dvm":
        raise IdentityUnavailable("NPU_BACKEND_OUT_OF_SCOPE:" + backend)
    import torch
    import torch_npu
    from torch_npu.npu.utils import get_cann_version
    from torch_npu import version

    if not devices:
        raise IdentityUnavailable("NPU_TARGET_UNKNOWN")
    if len(devices) != 1:
        raise IdentityUnavailable("NPU_MULTIPLE_TARGETS_UNSUPPORTED")
    try:
        targets = sorted({torch.npu.get_device_name(d) for d in devices})
        if any(not target for target in targets):
            raise IdentityUnavailable("NPU_TARGET_UNKNOWN")
        cann = {part: get_cann_version(part) for part in ("CANN", "RUNTIME", "COMPILER", "OPP")}
        if any(not value or str(value).lower() in ("unknown", "none") for value in cann.values()):
            raise IdentityUnavailable("NPU_CANN_IDENTITY_UNKNOWN")
        root = Path(torch_npu.__file__).parent
        result = {
            "schema": SCHEMA,
            "targets": targets,
            "torch_npu": {"version": version.__version__, "git": version.git_version,
                          "native": file_digest(root / "lib" / "libtorch_npu.so")},
            "cann": cann,
        }
        # DVM is linked into libtorch_npu; the Python builder is also codegen.
        result["dvm"] = {}
        for name in ("graph_build", "op_emitter", "mlir_fusion", "config", "decomp"):
            module = sys.modules.get("torch_npu._inductor.dvm." + name)
            path = getattr(module, "__file__", root / "_inductor" / "dvm" / (name + ".py"))
            result["dvm"][name] = file_digest(path)
        return canonical(result)
    except IdentityUnavailable:
        raise
    except (AttributeError, RuntimeError, OSError) as error:
        raise IdentityUnavailable(f"NPU_IDENTITY_UNAVAILABLE:{type(error).__name__}:{error}") from error
