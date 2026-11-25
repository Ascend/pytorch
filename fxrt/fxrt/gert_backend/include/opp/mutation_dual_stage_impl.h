#pragma once
#include "base/registry/op_impl_space_registry_v2.h"
namespace fx_converter {
bool InstallFxFillDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);
bool InstallFxCopyDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);
} // namespace fx_converter
