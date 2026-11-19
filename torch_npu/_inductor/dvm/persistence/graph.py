# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Graph-owned DVM assets: attach on compile, restore under the producer backend."""
from torch_npu.dynamo._package_compat.context import ACTIVE, preserve_device
from torch_npu._inductor.dvm.fx_asset import Uncacheable, materialize
from torch_npu.dynamo._package_compat.format import PackageCompatibilityError
from torch_npu.dynamo._package_compat.validation import bypass

from .validation import validate_dvm_assets


def attach_dvm_graph_assets(obj, graph):
    """Attach what a DVM graph needs in order to be restored elsewhere."""
    obj._npu_fx_assets = dict(getattr(graph, "_npu_fx_assets", {}))
    try:
        context = ACTIVE.get()
        if context is None:
            # Hashing already happened. A context created here would lose
            # the caller's options and authorize an inactive key.
            raise PackageCompatibilityError("NPU_REQUEST_UNAVAILABLE")
        if not context.persistable:
            raise PackageCompatibilityError(context.reason or "NPU_REQUEST_UNCACHEABLE")
        if not set(obj.device_idxs).issubset(device.index for device in context.devices):
            raise PackageCompatibilityError("NPU_TARGET_MISMATCH")
        obj._npu_manifest = context.manifest
        validate_dvm_assets(obj)
        obj._npu_persistable = True
    except PackageCompatibilityError as error:
        obj._npu_persistable = False
        obj._npu_uncacheable = Uncacheable()
        bypass(str(error))


def restore_dvm_graph_assets(obj, constants, original):
    """Restore graph-owned assets under the backend that produced them.

    ``original`` is the native ``after_deserialization``. The caller wraps this
    call in ``_NpuBackendScope(backend)`` so that the import of this module --
    which loads ``torch_npu._inductor`` and therefore a backend -- already
    happens under the producer's backend rather than the ambient one.
    """
    # Materialize, do not validate: this also runs while a package is
    # restored, where raising could not fall back to a miss.
    for asset in getattr(obj, "_npu_fx_assets", {}).values():
        materialize(asset)
    # The generated wrapper imports torch_npu._inductor, which loads
    # whichever backend the environment names. Publish the producer's.
    with preserve_device():
        return original(obj, constants)
