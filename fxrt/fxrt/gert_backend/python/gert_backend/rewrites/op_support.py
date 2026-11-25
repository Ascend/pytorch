"""Op-support hook: a call stays a logical Op only when the converter can lower it.

The supported set is the catalog (builtin rows and Python overlays) plus the
non-catalog names dedicated lowerings declare. Everything else becomes
Op.custom_call. This replaces fx_backend's check_op_support whitelist: membership
is the registry, not a second list of op names in Python.

FX_CONVERTER_FORCE_CUSTOM_CALL is a comma-separated list of logical op names
(for example ``sin,cos``). Those names are demoted even when the catalog can
lower them, so a test can exercise custom_call without removing the row.

A row is also demoted when GE would launch it through the aicore MIX path but the
runtime cannot hand out the hardware sync address that path needs
(``aclrtGetHardwareSyncAddr``). See ``_HARDWARE_SYNC_ADDRESS_OPS``.
"""

from __future__ import annotations

import logging
import os

import torch

from fxrt.ir import DataType, Op
from fxrt import fx_backend as fxb


# This module intentionally reaches into fx_backend internals for the converter-only
# hook overrides, under the same contract as the other rewrites modules.
# pylint: disable=protected-access


_logger = logging.getLogger(__name__)

_FORCE_CUSTOM_CALL_ENV = "FX_CONVERTER_FORCE_CUSTOM_CALL"

_CONVERTER_SUPPORTED_LOGICAL_OPS = None

# Catalog rows that GE lowers to an AscendC MIX kernel. The aicore launch path
# fetches the cross-core hardware sync address of such a kernel through
# aclrtGetHardwareSyncAddr and fails the whole run (UpdateEachArgsInfo, GE status
# 1343225857) where the runtime rejects that call with
# ACL_ERROR_RT_FEATURE_NOT_SUPPORT. On those devices the row runs through
# custom_call (the registered torch fallback) instead.
_HARDWARE_SYNC_ADDRESS_OPS = frozenset({"topk", "cumsum_ext"})

_HARDWARE_SYNC_ADDRESS_SUPPORTED = None


def _hardware_sync_addr_supported() -> bool:
    """Whether aclrtGetHardwareSyncAddr works on this device; probed once.

    The first graph build already runs with the NPU device selected, and the probe
    answers with the same status the GE launch path gets. A missing library or a
    failing probe assumes support, which keeps the catalog lowering.
    """
    global _HARDWARE_SYNC_ADDRESS_SUPPORTED  # pylint: disable=global-statement
    if _HARDWARE_SYNC_ADDRESS_SUPPORTED is None:
        try:
            import ctypes  # pylint: disable=import-outside-toplevel

            acl = ctypes.CDLL("libascendcl.so")
            query = acl.aclrtGetHardwareSyncAddr
            query.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            query.restype = ctypes.c_int
            address = ctypes.c_void_p()
            _HARDWARE_SYNC_ADDRESS_SUPPORTED = query(ctypes.byref(address)) == 0
        except Exception:  # pylint: disable=broad-except
            _HARDWARE_SYNC_ADDRESS_SUPPORTED = True
    return _HARDWARE_SYNC_ADDRESS_SUPPORTED


def _environment_forced_names():
    """Logical names listed in FX_CONVERTER_FORCE_CUSTOM_CALL, read on each build."""
    raw = os.environ.get(_FORCE_CUSTOM_CALL_ENV, "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _forced_custom_call_names():
    """Logical names demoted to custom_call for this build.

    The environment list always applies. Rows whose MIX-kernel launch needs a
    hardware sync address join it when the runtime cannot provide one.
    """
    names = _environment_forced_names()
    if not _hardware_sync_addr_supported():
        names = names | _HARDWARE_SYNC_ADDRESS_OPS
    return names


def _refresh_converter_supported_logical_ops() -> None:
    """Reload catalog and dedicated-lowering names. The next graph build uses them."""
    global _CONVERTER_SUPPORTED_LOGICAL_OPS  # pylint: disable=global-statement
    from fxrt.gert_backend import _fxrt_fx_converter  # pylint: disable=import-outside-toplevel

    _CONVERTER_SUPPORTED_LOGICAL_OPS = frozenset(_fxrt_fx_converter.supported_logical_ops())


def _converter_supported_logical_ops():
    """Logical names Convert can lower (catalog, Python overlays, dedicated lowerings)."""
    if _CONVERTER_SUPPORTED_LOGICAL_OPS is None:
        _refresh_converter_supported_logical_ops()
    return _CONVERTER_SUPPORTED_LOGICAL_OPS


def _has_symbolic_stride(fx_node) -> bool:
    """True when the FX node's output carries a symbolic (SymInt) stride.

    The builtin ``empty_strided`` lowering insists the requested strides are provably
    contiguous. Under ``torch.compile(dynamic=True)`` the stride list can contain SymInt
    expressions (e.g. ``128 * s74`` or a symbolic stride on a size-1 dim), which the C++
    lowering rejects with "symbolic stride is not provably contiguous". Such nodes are
    routed to ``custom_call`` (i.e. ``torch.empty_strided`` at runtime) instead.
    """
    if fx_node is None:
        return True
    value = fx_node.meta.get("example_value", None)
    if not isinstance(value, torch.Tensor):
        return True
    try:
        return any(isinstance(stride, torch.SymInt) for stride in value.stride())
    except Exception:  # pylint: disable=broad-exception-caught
        return True


def _converter_op_support_hook(op, output_value, input_nodes, fx_node=None):
    """Keep catalog and dedicated-lowering names; demote everything else to custom_call."""
    del input_nodes

    def fallback():
        if fx_node is not None:
            fx_node.meta["_fxrt_converter_custom_call"] = True
        return Op.custom_call

    if op == Op.custom_call:
        return fallback()
    if not hasattr(op, "name"):
        return op
    if op.name in _forced_custom_call_names():
        if op.name in _environment_forced_names():
            _logger.info("Op %s forced to custom_call by %s", op.name, _FORCE_CUSTOM_CALL_ENV)
        else:
            _logger.info(
                "Op %s demoted to custom_call: aclrtGetHardwareSyncAddr is unsupported",
                op.name,
            )
        return fallback()
    if op.name == "empty_strided":
        # When dynamic=True, the stride of empty_strided may be a SymInt expression.
        # Because C++ lowering cannot prove its contiguity, it raises the error
        # "symbolic stride is not provably contiguous". In this case, it falls back to
        # custom_call (executing torch.empty_strided at runtime), while static strides
        # still go through the native lowering path.
        if _has_symbolic_stride(fx_node):
            _logger.info("Op empty_strided has symbolic stride, fallback to custom_call")
            return fallback()
        return op
    if op.name == "empty":
        # The native allocation path constructs an ND contiguous tensor. Keep
        # CPU, pinned, non-strided and channels-last factories on Torch fallback.
        from fxrt.compiled_kernel_adapter import _layout_is_plain
        from fxrt.gert_backend.rewrites._common import example_of

        value = example_of(fx_node)
        if (not isinstance(value, torch.Tensor) or value.device.type != "npu" or
                value.layout != torch.strided or not _layout_is_plain(value) or
                fx_node.kwargs.get("pin_memory", False) or
                fx_node.kwargs.get("memory_format") not in (None, torch.contiguous_format)):
            return fallback()
    if op.name == "zeros":
        # Fill's native value/output dtypes. Preserve fallback for types whose
        # CANN binary or scalar materialization is unavailable.
        if output_value is None or not output_value.is_tensor():
            return fallback()
        dtype = output_value.to_tensor().dtype.value
        if dtype not in (DataType.Type.Float16, DataType.Type.BFloat16, DataType.Type.Float32,
                         DataType.Type.Int8, DataType.Type.Int32, DataType.Type.Int64, DataType.Type.Bool):
            return fallback()
    if op.name in _converter_supported_logical_ops():
        return op
    _logger.info("Op %s not supported by fx_converter, fallback to custom_call", op.name)
    return fallback()


def register() -> None:
    """Install the demotion hook and refresh the supported-name set.

    Idempotent. Called on each graph build so a catalog overlay registered since the
    previous build is included. Stays for the rest of the process.
    """
    fxb._CONVERTER_GRAPH_BUILD = True
    _refresh_converter_supported_logical_ops()
    if hasattr(fxb, "register_op_support_hook"):
        fxb.register_op_support_hook(_converter_op_support_hook)
    else:
        fxb._check_and_fallback_op_by_backend_support = _converter_op_support_hook
