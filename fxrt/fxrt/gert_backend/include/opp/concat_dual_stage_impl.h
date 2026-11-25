#pragma once

#include "base/registry/op_impl_space_registry_v2.h"

namespace fx_converter {

// Install Fx-owned dual-stage prepare/launch for OPP type "Concat" (aclnnCat wrapper).
// Catalog rows cat/concat lower onto this type.
bool InstallFxConcatDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);

} // namespace fx_converter
