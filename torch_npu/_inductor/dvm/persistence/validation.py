# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Can a DVM graph record be executed after it is restored?

Format and scope checks only. *Which* options produced the artifact is not
re-examined here, so a graph from an older configuration stays usable, as on
CUDA/XPU; only what makes the bytes executable is verified.
"""
from torch_npu._inductor.dvm.fx_asset import (
    FxAsset, LiveFxAsset, _parse_source, check_no_triton_source,
)
from torch_npu.dynamo._package_compat.format import Manifest, PackageCompatibilityError
from torch_npu.dynamo._package_compat.validation import walk_objects


def check_dvm_graph(graph):
    """Format and scope gate for one DVM graph we own."""
    manifest = getattr(graph, "_npu_manifest", None)
    if not isinstance(manifest, Manifest):
        raise PackageCompatibilityError("NPU_MANIFEST_MISSING")
    manifest.check_supported("dvm")
    bundle = getattr(graph, "_triton_bundle", None)
    if bundle and getattr(bundle, "static_autotuners", ()):
        raise PackageCompatibilityError("NPU_STATIC_TRITON_OUT_OF_SCOPE")
    # Transitional mixed-graph support: leave ordinary Triton kernels and their
    # bundles to upstream. No extra Triton identity/config invalidation is added;
    # reuse requires an unchanged Triton toolchain and relevant configuration.
    _parse_source(graph.source_code)


def validate_dvm_assets(backend):
    """Verify that every DVM record can be executed after restoring it."""
    from torch._inductor.output_code import CompiledFxGraph

    for item in walk_objects(backend):
        if isinstance(item, LiveFxAsset):
            raise PackageCompatibilityError("NPU_FX_ASSET_NOT_PORTABLE")
        if getattr(item, "_npu_persistable", True) is False:
            raise PackageCompatibilityError("NPU_ARTIFACT_NOT_PORTABLE")
        if isinstance(item, CompiledFxGraph):
            if "npu" in item.device_types:
                check_dvm_graph(item)
            assets = getattr(item, "_npu_fx_assets", {})
            for key, asset in assets.items():
                if isinstance(asset, LiveFxAsset):
                    raise PackageCompatibilityError("NPU_FX_ASSET_NOT_PORTABLE")
                if not isinstance(asset, FxAsset):
                    raise PackageCompatibilityError("NPU_FX_ASSET_TYPE_INVALID")
                metadata = asset.validate(key)
                if "npu" in item.device_types:
                    for module in metadata["modules"]:
                        if module.get("kind") == "graph":
                            check_no_triton_source("\n".join(module["imports"]) + "\n" + module["code"])
            # Exact dependencies are also embedded as IDs in the generated wrapper.
            import re

            references = set(re.findall(r"['\"]npu_fx_asset_id['\"]\s*:\s*['\"]([a-z0-9_]+)['\"]",
                                        item.source_code))
            if not references.issubset(assets):
                raise PackageCompatibilityError("NPU_FX_ASSET_INCOMPLETE")
