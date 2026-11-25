#pragma once

#include "base/registry/op_impl_space_registry_v2.h"

namespace fx_converter {

// Use aclnnBitwiseNot for OPP "Invert", preserving the full input/output dtype.
bool InstallFxBitwiseNotDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);

} // namespace fx_converter
