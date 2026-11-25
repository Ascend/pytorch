#pragma once

#include "base/registry/op_impl_space_registry_v2.h"

namespace fx_converter {

// Install Fx-owned dual-stage prepare/launch for OPP type "GatherElements" (aclnnGather).
// Path C GatherElementsV2 bins lack bool / int64-index; OPP host has no prepare/launch → Path B.
bool InstallFxGatherDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);

} // namespace fx_converter
