# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Package entries owned by a DVM compile, made portable for native installation."""
import copy

from torch_npu._compat.version import CURRENT_VERSION
from torch_npu.dynamo._package_compat.format import PackageCompatibilityError
from torch_npu.dynamo._package_compat.validation import (
    bypass, has_dvm_records, restorable_backends,
)

from .validation import validate_dvm_assets


def label_dvm_package_entry(entry, package):
    """Label a DVM-owned package with the device type it was produced on."""
    # Native labels a package from its first graph node and stops at the first
    # non-tensor value, so a graph that starts from a CPU constant is labelled
    # "cpu". check_compatibility only takes its device branch for a listed
    # device type, so this label is what lets the NPU environment check run.
    entry.device_type = "npu"
    # Keep npu on the package itself too, or a later cache_entry() call
    # (save_package calls it twice) would relabel the package "cpu" again.
    # COMPAT(< 2.15): upstream pytorch#196814 (first in torch 2.15) replaced
    # CompilePackage._device_type (str) with _device_types (frozenset), which
    # cache_entry() folds through _collapse_device_types; 2.13/2.14 record the
    # single native string, and update_device_type keeps it across calls the
    # same way.
    # CAN REMOVE the else branch when MIN_SUPPORTED_VERSION >= (2, 15)
    if CURRENT_VERSION >= (2, 15):
        package._device_types |= {"npu"}
    else:
        package._device_type = "npu"


def portable_dvm_entries(entries):
    """Drop or trim the DVM entries native installation could not restore."""
    portable = {}
    for key, entry in entries.items():
        if has_dvm_records(entry):
            # Native installation skips every bypassed code and would still
            # report a cache hit, so an entry with no live code is empty, not
            # reusable.
            if not any(not code.bypassed for code in entry.dynamo.codes):
                bypass("NPU_PACKAGE_EMPTY")
                continue
            try:
                required = restorable_backends(entry)
                validate_dvm_assets(required)
            except PackageCompatibilityError as error:
                bypass(str(error))
                continue
            if set(required) != set(entry.backends):
                # A bypassed segment may have left behind a found but unusable
                # artifact. It is never restored, and keeping it would let one
                # dead segment make this write fail.
                entry = copy.copy(entry)
                entry.backends = required
        portable[key] = entry
    return portable
