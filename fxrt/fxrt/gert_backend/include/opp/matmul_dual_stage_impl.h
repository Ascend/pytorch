#pragma once

#include "base/registry/op_impl_space_registry_v2.h"

namespace fx_converter {

// OPP MatMul/BatchMatMul dual-stage calls aclnnMm without applying transpose_x / adj.
// Path B wraps aclnnMm / aclnnBatchMatMul on a last-two transposed ND view.
bool InstallFxMatMulDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);
bool InstallFxBatchMatMulDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);
bool InstallFxAddmmDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);

} // namespace fx_converter
