#pragma once

#include "base/registry/op_impl_space_registry_v2.h"

namespace fx_converter {

// Install prepare/launch for FlashAttentionScore. The call is aclnnFlashAttentionScore,
// with softmax_max, softmax_sum, softmax_out, attention_out as the four outputs.
bool InstallFxFlashAttentionScoreDualStage(const gert::OpImplSpaceRegistryV2Ptr& registry);

} // namespace fx_converter
