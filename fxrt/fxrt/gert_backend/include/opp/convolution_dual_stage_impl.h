#pragma once

#include "base/registry/op_impl_space_registry_v2.h"

namespace fx_converter {

// Fx-owned dual-stage Convolution.
//
// OPP registers no "Convolution" OpImpl entry (only legacy Conv2D/Conv3D tiling),
// so there is nothing to patch under that name. The catalog row therefore reuses
// the existing "Conv2D" entry as a mount point and patches its prepare/launch with
// aclnnConvolution, which is the identical kernel torch_npu calls for
// aten.convolution -- mode3 must stay bit-for-bit with mode1.
bool InstallFxConvolutionDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);

} // namespace fx_converter
