# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""DVM-owned bodies for compile-package and cache persistence.

The public adapters in :mod:`torch_npu.dynamo._package_compat` keep only the
routing decision -- "is this a DVM request/artifact?" -- and hand the work to
this package. The import is lazy at every call site: importing
``torch_npu._inductor.dvm`` loads the DVM C extension and queries the SoC
version, and a cache load must not trigger that before a backend is selected
(see the independence rule in ``_package_compat/__init__.py``).

Layout, by concern:

``request``
    Capturing one compilation's producer identity and freezing it for the
    duration of the compile (:func:`make_context`, :func:`request_scope`, ...).
``identity``
    The values that identity is built from: DVM/ANIR config projection and the
    compiler-installation fingerprint.
``validation``
    "Can these bytes be executed after restore?" for a DVM graph record.
``graph``
    Attaching and restoring graph-owned assets.
``entries``
    Labelling and trimming package entries for native installation.
``cache_identity``
    The no-op pass whose UUID publishes the identity into FX/AOT cache keys.

The public layer imports nothing from here at module scope; every caller is
inside a function, so a non-DVM compile never loads this package. Types that
exist only on this side (``CompileRequest``, ``RequestContext``) keep the names
they had, and names that could be confused with a public helper carry a ``dvm``
prefix. Another backend that needs the same cache adaptation should build a
parallel package instead of adding branches here.
"""
from .cache_identity import DvmCacheIdentity
from .entries import label_dvm_package_entry, portable_dvm_entries
from .graph import attach_dvm_graph_assets, restore_dvm_graph_assets
from .identity import collect_identity, initialization_state, record_initialization
from .request import (
    CompileRequest,
    RequestContext,
    bind_inner_compile,
    capture_configs,
    configuration_scope,
    make_context,
    request_scope,
)
from .validation import check_dvm_graph, validate_dvm_assets

__all__ = [
    "CompileRequest",
    "DvmCacheIdentity",
    "RequestContext",
    "attach_dvm_graph_assets",
    "bind_inner_compile",
    "capture_configs",
    "check_dvm_graph",
    "collect_identity",
    "configuration_scope",
    "initialization_state",
    "label_dvm_package_entry",
    "make_context",
    "portable_dvm_entries",
    "record_initialization",
    "request_scope",
    "restore_dvm_graph_assets",
    "validate_dvm_assets",
]
