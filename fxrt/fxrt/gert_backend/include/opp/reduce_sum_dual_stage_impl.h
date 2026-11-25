#pragma once

#include "base/registry/op_impl_space_registry_v2.h"

namespace fx_converter {

// Install Fx-owned dual-stage prepare/launch for OPP type "ReduceSum" (aclnnReduceSum).
// Path C has no BOOL binary; OPP host has no prepare/launch → Path B aclnn.
bool InstallFxReduceSumDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);

} // namespace fx_converter
