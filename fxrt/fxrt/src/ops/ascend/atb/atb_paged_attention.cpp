#include "ops/ascend/atb/atb_paged_attention.h"

#include "ops/op_register.h"

namespace fxrt {
namespace ops {

AtbPagedAttention::AtbPagedAttention() : AtbBase("paged_attention") {}

namespace {
// Match PagedAttentionAtb.cpp signature:
// (query, key_cache, value_cache, num_kv_heads, num_heads, scale_value,
//  block_table, context_lens, out, workspace)
constexpr size_t kQueryIdx = 0;
constexpr size_t kKeyCacheIdx = 1;
constexpr size_t kValueCacheIdx = 2;
constexpr size_t kNumKvHeadsIdx = 3;
constexpr size_t kNumHeadsIdx = 4;
constexpr size_t kScaleValueIdx = 5;
constexpr size_t kBlockTableIdx = 6;
constexpr size_t kContextLensIdx = 7;
constexpr size_t kOutIdx = 8;
constexpr size_t kWorkspaceIdx = 9;
} // namespace

static OpsErrorCode GetPagedAttentionParam(
    const std::vector<const ir::Value*>& inputs,
    const ir::Value* output,
    atb::infer::PagedAttentionParam* param) {
  CHECK_IF_NULL(param);
  // At least 10 inputs:
  // query, key_cache, value_cache, num_kv_heads, num_heads, scale_value,
  // block_table, context_lens, out, workspace
  if (inputs.size() < kWorkspaceIdx + 1) {
    RT_GLOG(ERROR) << "Invalid parameters for AtbPagedAttention, input size: " << inputs.size();
    return OpsErrorCode::INVALID_PARAM;
  }
  // Read scalar parameters: num_kv_heads, num_heads, scale_value.
  const int64_t num_kv_heads = inputs[kNumKvHeadsIdx]->ToInt();
  const int64_t num_heads = inputs[kNumHeadsIdx]->ToInt();
  const double scale_value = inputs[kScaleValueIdx]->ToDouble();

  using PagedAttentionParam = atb::infer::PagedAttentionParam;
  PagedAttentionParam paged_param;
  paged_param.headNum = static_cast<int32_t>(num_heads);
  // qkScale uses scale_value from front-end.
  paged_param.qkScale = static_cast<float>(scale_value);
  paged_param.kvHeadNum = static_cast<int32_t>(num_kv_heads);
  paged_param.maskType = PagedAttentionParam::UNDEFINED;
  paged_param.batchRunStatusEnable = false;
  paged_param.quantType = PagedAttentionParam::TYPE_QUANT_UNDEFINED;
  paged_param.outDataType = ACL_DT_UNDEFINED;
  paged_param.hasQuantOffset = false;
  paged_param.compressType = PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
  paged_param.calcType = PagedAttentionParam::CALC_TYPE_UNDEFINED;
  paged_param.scaleType = PagedAttentionParam::SCALE_TYPE_TOR;
  paged_param.inputLayout = atb::infer::TYPE_BSND;
  paged_param.mlaVHeadSize = 0;

  *param = paged_param;
  (void)output;
  return OpsErrorCode::SUCCESS;
}

OpsErrorCode AtbPagedAttention::CalcWorkspace(
    const std::vector<const ir::Value*>& inputs,
    const ir::Value* output,
    size_t* workspace_size) {
  CHECK_IF_NULL(workspace_size);
  // Real output tensor is passed in via inputs[kOutIdx], consistent with PagedAttentionAtb.cpp.
  CHECK_IF_NULL(inputs[kOutIdx]);
  auto* real_output = const_cast<ir::Value*>(inputs[kOutIdx]);

  auto old_hash = current_hash_id_;
  atb::infer::PagedAttentionParam param;
  auto ret = GetPagedAttentionParam(inputs, output, &param);
  if (ret != OpsErrorCode::SUCCESS) {
    return ret;
  }

  auto& entry = GetOrCreateEntry(param, inputs, real_output);
  if (old_hash != current_hash_id_) {
    // Inputs to ATB kernel: query, key_cache, value_cache, block_table, context_lens
    param_setter_.SetIndex({kQueryIdx, kKeyCacheIdx, kValueCacheIdx, kBlockTableIdx, kContextLensIdx}, {0})
        .Input(inputs[kQueryIdx])
        .Input(inputs[kKeyCacheIdx])
        .Input(inputs[kValueCacheIdx])
        .Input(inputs[kBlockTableIdx])
        .Input(inputs[kContextLensIdx])
        .Output(real_output);
  }
  param_setter_.Update(inputs, real_output);
  return GetWorkspaceSize(entry, param_setter_.variant_pack, workspace_size);
}

OpsErrorCode AtbPagedAttention::Launch(
    const std::vector<const ir::Value*>& inputs,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  CHECK_IF_NULL(stream);
  return LaunchAtb(param_setter_.variant_pack, workspace, workspaceSize, static_cast<aclrtStream>(stream));
}

FXRT_REG_OP(paged_attention, AtbPagedAttention, Ascend);

} // namespace ops
} // namespace fxrt
