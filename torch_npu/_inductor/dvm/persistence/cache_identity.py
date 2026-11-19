# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Publish the effective DVM request through Inductor's backend pass UUID."""
from torch._inductor.custom_graph_pass import CustomGraphModulePass

from torch_npu.dynamo._package_compat.context import ACTIVE
from torch_npu.dynamo._package_compat.format import PackageCompatibilityError


class DvmCacheIdentity(CustomGraphModulePass):
    """A no-op codegen pass whose UUID carries the owning request's identity.

    Native FX and AOT hashes include every registered backend's pass UUID, even
    for CPU graphs, so this is how a DVM compilation's environment and codegen
    knobs reach the *lower* cache keys. That keeps working exactly as before.

    Note what an option change does and does not do after the alignment with
    native packages: it changes these keys, so graphs are recompiled, but it no
    longer rejects a saved compile package. Reuse of a package is decided by the
    environment only, which is the CUDA/XPU behaviour.

    Outside a DVM request use a stable neutral UUID; never probe devices or capture
    another caller's global configuration. Other backend activations use their
    original registration and clear this custom pass.
    """

    def __call__(self, gm):
        pass

    def uuid(self):
        from torch._inductor.codecache import BypassFxGraphCache

        context = ACTIVE.get()
        if context is None:
            return ("torch_npu.dvm", "inactive")
        try:
            if not context.persistable:
                raise PackageCompatibilityError(
                    "NPU_REQUEST_UNCACHEABLE:" + str(context.reason or "identity unavailable"))
            context.manifest.check_supported("dvm")
            return ("torch_npu.dvm", context.manifest.schema,
                    context.manifest.request_json, context.manifest.identity_json)
        except PackageCompatibilityError as error:
            raise BypassFxGraphCache(str(error)) from error
