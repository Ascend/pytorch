# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""DVM FX fallback archives: real constants, storage aliases and graph-owned assets.

Archives contain generated Python, just like Inductor caches. They are trusted
compiler artifacts, not a sandbox or a format for loading untrusted models.
"""
import ast
import contextlib
import dataclasses
import errno
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
import threading
import weakref

from torch_npu.dynamo._package_compat.format import PackageCompatibilityError, json_bytes

FX_SCHEMA = 1
_LIVE = weakref.WeakValueDictionary()
_LOCK = threading.RLock()
_CONTAINER_TYPES = frozenset({"Module", "ModuleList", "ModuleDict", "Sequential"})


def _parse_source(source):
    try:
        return ast.parse(source)
    except (SyntaxError, TypeError) as error:
        raise PackageCompatibilityError(f"NPU_SOURCE_UNSUPPORTED:{type(error).__name__}:{error}") from error


def check_no_triton_source(source):
    """Keep Triton dependencies out of our FX fallback archive.

    Native generated wrappers/bundles may contain Triton, but this archive does
    not restore its runtime side tables. AST inspection ignores comments/strings;
    this is a support boundary for trusted generated code, not a Python sandbox.
    """
    tree = _parse_source(source)
    kernel_ops = {"triton", "triton_kernel_wrapper_functional", "triton_kernel_wrapper_mutation", "wrap_triton"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in kernel_ops:
            raise PackageCompatibilityError("NPU_TRITON_OUT_OF_SCOPE")
        if isinstance(node, ast.Name) and node.id in kernel_ops:
            raise PackageCompatibilityError("NPU_TRITON_OUT_OF_SCOPE")
        names = ()
        if isinstance(node, ast.Import):
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names = (node.module or "",)
        if any(name == "triton" or name.startswith("triton.")
               or name == "torch._higher_order_ops.triton_kernel_wrap" for name in names):
            raise PackageCompatibilityError("NPU_TRITON_OUT_OF_SCOPE")


class UnsupportedFxConstant(RuntimeError):
    pass


class UnresolvedFxConstant(RuntimeError):
    """No real value exists: neither persistence nor eager fallback can invent it."""


class Uncacheable:
    """Let native cache saves handle nonportable artifacts as pickle failures."""

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        raise pickle.PicklingError("NPU_ARTIFACT_NOT_PORTABLE")


def _recoverable_resource_error(error):
    import torch

    return (isinstance(error, (MemoryError, torch.OutOfMemoryError))
            or isinstance(error, OSError) and error.errno in
            (errno.ENOSPC, errno.EDQUOT, errno.ENOMEM, errno.EFBIG))


@contextlib.contextmanager
def real_tensor_mode():
    import torch
    from torch._dispatch.python import no_python_dispatcher
    from torch._subclasses.fake_tensor import unset_fake_temporarily
    from torch.utils._python_dispatch import _disable_current_modes

    with torch._guards.tracing(None), unset_fake_temporarily(), _disable_current_modes(), no_python_dispatcher():
        yield


@dataclasses.dataclass(frozen=True)
class FxAsset:
    metadata: bytes
    payload: bytes

    @functools.cached_property
    def asset_id(self):
        digest = hashlib.sha256()
        for part in (self.metadata, self.payload):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
        return digest.hexdigest()

    def __reduce__(self):
        # Do not persist a cached digest. Each deserialized asset verifies its
        # immutable bytes once, then shares that result across restore checks.
        return type(self), (self.metadata, self.payload)

    def validate(self, expected_id=None):
        if not isinstance(self.metadata, bytes) or not isinstance(self.payload, bytes):
            raise PackageCompatibilityError("NPU_FX_ASSET_FORMAT_UNSUPPORTED")
        actual_id = self.asset_id
        if expected_id is not None and (not re.fullmatch(r"[0-9a-f]{64}", expected_id) or actual_id != expected_id):
            raise PackageCompatibilityError("NPU_FX_ASSET_DIGEST_MISMATCH")
        meta = json.loads(self.metadata)
        if not isinstance(meta, dict):
            # json.loads returns any JSON value; a dict is the only shape export writes.
            raise PackageCompatibilityError("NPU_FX_ASSET_FORMAT_UNSUPPORTED")
        if meta.get("schema") != FX_SCHEMA:
            raise PackageCompatibilityError("NPU_FX_ASSET_SCHEMA_UNSUPPORTED")
        if meta.get("format") != "torch_save":
            raise PackageCompatibilityError("NPU_FX_ASSET_FORMAT_UNSUPPORTED")
        for module in meta["modules"]:
            if "alias" in module or module.get("kind") == "graph":
                continue
            if module.get("kind") == "container" and module.get("type") in _CONTAINER_TYPES:
                continue
            raise PackageCompatibilityError("NPU_FX_MODULE_UNSUPPORTED")
        return meta


class LiveFxAsset:
    """Correct ordinary execution for unsupported objects; never persist them."""
    def __init__(self, gm):
        import uuid

        self.asset_id = "live_" + uuid.uuid4().hex
        self.gm = gm

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        raise pickle.PicklingError("NPU_FX_ASSET_NOT_PORTABLE")


class _StatePickler(pickle.Pickler):
    """Use native untyped-storage reducers, including for legacy dtypes.

    torch.save's typed-storage path rejects different dtypes sharing storage.
    Selecting the native v3 reducer preserves that case without copying or
    rebuilding storages ourselves. This Pickler is local to this archive only.
    """
    def reducer_override(self, value):
        import torch

        if type(value) is torch.Tensor:
            # As with compiler constants generally, transient Python attributes
            # on Tensor objects are not part of the executable archive.
            rebuild, args = value._reduce_ex_internal(2)
            if rebuild is torch._utils._rebuild_tensor_v2:
                return (torch._utils._rebuild_tensor_v3,
                        (args[0]._untyped_storage, *args[1:6], value.dtype, *args[6:]))
            return rebuild, args
        return NotImplemented


class _StatePickle:
    Pickler = _StatePickler


def _check_value(value, where):
    import torch
    from torch._subclasses.fake_tensor import FakeTensor

    if value is None or type(value) in (str, int, float, bool):
        return
    if isinstance(value, torch.Tensor):
        if isinstance(value, FakeTensor) or value.is_meta:
            kind = "a FakeTensor" if isinstance(value, FakeTensor) else "a meta tensor"
            raise UnresolvedFxConstant(
                f"NPU_FX_CONSTANT_HAS_NO_REAL_VALUE: {where} is {kind}; "
                "the FX fallback has no real value to execute with"
            )
        if value.layout != torch.strided or value.is_quantized or value.is_conj() or value.is_neg():
            raise UnsupportedFxConstant("FX constant needs a real supported strided storage")
        if type(value) not in (torch.Tensor, torch.nn.Parameter):
            raise UnsupportedFxConstant("tensor subclasses need an explicit FX archive adapter")
        if value.device.type == "npu":
            import torch_npu

            if torch_npu.get_npu_format(value) not in (0, 2):
                raise UnsupportedFxConstant("non-base NPU storage needs an explicit format adapter")
        return
    if isinstance(value, (torch.dtype, torch.device)):
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _check_value(item, where)
        return
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for item in value.values():
            _check_value(item, where)
        return
    raise UnsupportedFxConstant(f"unsupported FX attribute: {type(value).__qualname__}")


def _module_spec(module):
    import torch
    from torch.fx.graph_module import _format_import_statement
    from torch.package import sys_importer

    if isinstance(module, torch.fx.GraphModule):
        python_code = module.graph.python_code("self")
        try:
            check_no_triton_source(module.code)
        except PackageCompatibilityError as error:
            # A live fallback retains the real graph/side table for this process;
            # never export an asset which secretly depends on a Triton kernel.
            raise UnsupportedFxConstant(f"FX graph is outside DVM persistence scope: {error}") from error
        if any(getattr(value, "__module__", None) == "__main__" for value in python_code.globals.values()):
            raise UnsupportedFxConstant("FX globals from __main__ are not portable")
        try:
            imports = [_format_import_statement(k, v, sys_importer) for k, v in python_code.globals.items()]
        except (ImportError, AttributeError, RuntimeError) as error:
            raise UnsupportedFxConstant(f"FX globals are not importable: {type(error).__name__}:{error}") from error
        try:
            check_no_triton_source("\n".join(imports))
        except PackageCompatibilityError as error:
            raise UnsupportedFxConstant(f"FX imports are outside DVM persistence scope: {error}") from error
        return {"kind": "graph", "code": module.code, "imports": imports}
    if type(module) in (torch.nn.Module, torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Sequential):
        return {"kind": "container", "type": type(module).__name__}
    # DVM make_fx kernels carry operator graphs and get_attr state. Arbitrary
    # builtin call_module nodes remain executable as LiveFxAsset, not archives.
    raise UnsupportedFxConstant(f"unsupported FX child module: {type(module).__qualname__}")


def export_fx(gm):
    """Archive get_attr/module state only. Runtime placeholders remain inputs."""
    import torch

    modules, states, module_ids = [], {}, {}
    required_attributes = {}
    for child in gm.modules():
        if not isinstance(child, torch.fx.GraphModule):
            continue
        for node in child.graph.nodes:
            if node.op != "get_attr":
                continue
            parts = str(node.target).split(".")
            parent = child
            for part in parts[:-1]:
                parent = getattr(parent, part)
                if not isinstance(parent, torch.nn.Module):
                    raise UnsupportedFxConstant("get_attr traverses a non-module object")
            required_attributes.setdefault(id(parent), set()).add(parts[-1])

    def visit(module, path):
        if id(module) in module_ids:
            modules.append({"path": path, "alias": module_ids[id(module)]})
            return
        module_ids[id(module)] = path
        spec = dict(_module_spec(module), path=path)
        state = {"training": module.training, "_parameters": module._parameters,
                 "_buffers": module._buffers,
                 "_non_persistent_buffers_set": sorted(module._non_persistent_buffers_set)}
        for name, value in (*module._parameters.items(), *module._buffers.items()):
            _check_value(value, f"{path}.{name}" if path else name)
        for name in sorted(required_attributes.get(id(module), ())):
            if name in module._parameters or name in module._buffers or name in module._modules:
                continue
            value = getattr(module, name)
            _check_value(value, f"{path}.{name}" if path else name)
            state[name] = value
        states[path] = state
        modules.append(spec)
        # named_children() deduplicates aliases; _modules preserves them.
        for name, child in module._modules.items():
            if child is None:
                spec.setdefault("none_children", []).append(name)
            else:
                visit(child, path + "." + name if path else name)

    with real_tensor_mode():
        visit(gm, "")
        buffer = io.BytesIO()
        # Serialize all states together: native memoization preserves aliases
        # across modules as well as across attributes. Never save a GraphModule.
        torch.save(states, buffer, pickle_module=_StatePickle, pickle_protocol=2)
    metadata = json_bytes({"schema": FX_SCHEMA, "format": "torch_save", "modules": modules})
    return FxAsset(metadata, buffer.getvalue())


def restore_fx(asset, *, expected_id=None):
    import torch

    meta = asset.validate(expected_id)
    asset_id = asset.asset_id
    with real_tensor_mode(), torch.no_grad():
        states = torch.load(io.BytesIO(asset.payload), weights_only=True)
        paths = {spec["path"] for spec in meta["modules"] if "alias" not in spec}
        if not isinstance(states, dict) or states.keys() != paths:
            raise PackageCompatibilityError("NPU_FX_ASSET_INCOMPLETE")
        built = {}
        for spec in meta["modules"]:
            path = spec["path"]
            if "alias" in spec:
                module = built[spec["alias"]]
            else:
                if spec["kind"] == "graph":
                    source = "\n".join(spec["imports"]) + "\nclass NpuFxModule(torch.nn.Module):\n"
                    source += "\n".join("    " + line for line in spec["code"].splitlines()) + "\n"
                    namespace = {"torch": torch, "__name__": "_torch_npu_fx_" + asset_id}
                    exec(compile(source, "<torch_npu_fx_" + asset_id + ">", "exec"), namespace)
                    module = namespace["NpuFxModule"]()
                elif spec["kind"] == "container":
                    module = getattr(torch.nn, spec["type"])()
                else:
                    raise PackageCompatibilityError("NPU_FX_MODULE_INVALID")
                state = states[path]
                state["_non_persistent_buffers_set"] = set(state["_non_persistent_buffers_set"])
                module.__setstate__(state)
                for name in spec.get("none_children", ()):
                    module.add_module(name, None)
            built[path] = module
            if path:
                parent, _, name = path.rpartition(".")
                built[parent].add_module(name, module)
    return built[""]


def _asset_path(asset_id):
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    if not re.fullmatch(r"[0-9a-f]{64}", asset_id):
        raise PackageCompatibilityError("NPU_FX_ASSET_ID_INVALID")
    return Path(cache_dir()) / "npu_fx_assets" / asset_id / "asset"


def _read_asset(path):
    """Turn on-disk bytes into an asset, or report the entry as unusable.

    This is the one place that decides what "the cache entry is bad" means, so
    callers do not enumerate pickle's failure modes. A moved or renamed module,
    a truncated write and a stale schema all land in the same answer: rewrite.
    Resource exhaustion is the deliberate exception -- it is not a statement
    about the bytes, and the caller has a separate fallback for it.
    """
    try:
        with path.open("rb") as stream:
            asset = pickle.load(stream)
    except Exception as error:
        if _recoverable_resource_error(error):
            raise
        raise PackageCompatibilityError(
            f"NPU_FX_ASSET_UNREADABLE:{type(error).__name__}:{error}") from error
    if not isinstance(asset, FxAsset):
        raise PackageCompatibilityError("NPU_FX_ASSET_TYPE_INVALID")
    return asset


def materialize(asset):
    """Reuse verified local bytes; atomically replace missing or corrupt assets."""
    asset.validate()
    path = _asset_path(asset.asset_id)
    try:
        saved = _read_asset(path)
        # Compare actual bytes, not file timestamps: same-sized overwrites can
        # share a filesystem timestamp. Incoming bytes were already validated.
        if saved.metadata == asset.metadata and saved.payload == asset.payload:
            return path
    except (OSError, PackageCompatibilityError) as error:
        # A missing or damaged file is a miss we can rewrite. Exhausted
        # resources are the caller's decision: ``register_fx_asset`` degrades
        # to a live asset, and rewriting would only fail the same way.
        if _recoverable_resource_error(error):
            raise
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Same pickle format, but do not allocate another whole-asset bytes object.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".asset-", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(asset, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def register_fx_asset(gm, graph):
    """Capture ownership on GraphLowering, never by scanning a global cache."""
    try:
        asset = export_fx(gm)
        materialize(asset)
    except (MemoryError, OSError, RuntimeError) as error:
        if not isinstance(error, UnsupportedFxConstant) and not _recoverable_resource_error(error):
            raise
        import logging
        from torch._dynamo.utils import counters

        reason = f"{type(error).__name__}:{error}"
        logging.getLogger(__name__).warning("FX fallback remains live; package bypass: %s", reason)
        counters["npu_package"]["NPU_FX_PERSISTENCE_BYPASS"] += 1
        asset = LiveFxAsset(gm)
        asset.reason = reason
        with _LOCK:
            _LIVE[asset.asset_id] = asset
    if not hasattr(graph, "_npu_fx_assets"):
        graph._npu_fx_assets = {}
    graph._npu_fx_assets[asset.asset_id] = asset
    return asset.asset_id


def load_fx_asset(asset_id):
    if asset_id.startswith("live_"):
        with _LOCK:
            asset = _LIVE.get(asset_id)
        if asset is None:
            raise PackageCompatibilityError("NPU_LIVE_FX_ASSET_EXPIRED")
        return asset.gm
    path = _asset_path(asset_id)
    try:
        asset = _read_asset(path)
        return restore_fx(asset, expected_id=asset_id)
    except (OSError, PackageCompatibilityError) as error:
        # ``_read_asset`` has already folded every unusable-bytes case into
        # PackageCompatibilityError; this only adds the asset id for a reader
        # who has to find the file. ``restore_fx`` raises the same type for a
        # payload that does not match its metadata. Exhausted resources say
        # nothing about the bytes, so they keep their own type.
        if _recoverable_resource_error(error):
            raise
        raise PackageCompatibilityError(
            f"NPU_FX_ASSET_MISSING_OR_CORRUPT:{asset_id}:{type(error).__name__}:{error}"
        ) from error
