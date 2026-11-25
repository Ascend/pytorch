#pragma once

#include "base/registry/op_impl_space_registry_v2.h"

namespace fx_converter {

// Baddbmm mounts on OPP's BatchMatMulV2 entry and calls aclnnBaddbmm directly.
bool InstallFxBaddbmmDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);

} // namespace fx_converter
