#pragma once

#include "base/registry/op_impl_space_registry_v2.h"

namespace fx_converter {

// Install Fx-owned dual-stage prepare/launch for OPP type "ReduceMax" (aclnnAmax).
// The float32/int32 AiCore bin is MIX-only and this device cannot launch it.
bool InstallFxAmaxDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);

} // namespace fx_converter
