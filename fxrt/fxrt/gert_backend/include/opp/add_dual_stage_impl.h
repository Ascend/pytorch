#pragma once

#include "base/registry/op_impl_space_registry_v2.h"

namespace fx_converter {

// Install Fx-owned dual-stage prepare/launch for OPP type "Add" (aclnnAdd wrapper).
// Patches the loaded SpaceRegistry so DualStageOpLowering::Match accepts "Add".
bool InstallFxAddDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);

} // namespace fx_converter
