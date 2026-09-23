#include "ops/ascend/atb/atb_reshape_and_cache.h"
#include "ops/op_register.h"
#include "ops/utils/aten_convert.h"

namespace fxrt {
namespace ops {

AtbReshapeAndCache::AtbReshapeAndCache() : AtbBase("reshape_and_cache") {}

namespace {
constexpr size_t kKeyIdx = 0;
constexpr size_t kValueIdx = 1;
constexpr size_t kKeyCacheIdx = 2;
constexpr size_t kValueCacheIdx = 3;
constexpr size_t kSlotIndicesIdx = 4;
} // namespace

OpsErrorCode AtbReshapeAndCache::CalcWorkspace(
    const std::vector<const ir::Value*>& inputs,
    const ir::Value* output,
    size_t* workspace_size) {
  CHECK_IF_NULL(workspace_size);
  if (inputs.size() < 5) {
    RT_GLOG(ERROR) << "Invalid parameters for AtbReshapeAndCache::CalcWorkspace, input size: " << inputs.size();
    return OpsErrorCode::INVALID_PARAM;
  }
  auto old_hash = current_hash_id_;
  atb::infer::ReshapeAndCacheParam param;
  param.compressType = atb::infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED;

  auto key_cache_tensor = inputs[kKeyCacheIdx]->ToTensor();
  auto value_cache_tensor = inputs[kValueCacheIdx]->ToTensor();
  CHECK_IF_NULL(key_cache_tensor);
  CHECK_IF_NULL(value_cache_tensor);

  const auto key_cache_format = ConvertMemoryFormatToAclFormat(key_cache_tensor->Format());
  const auto value_cache_format = ConvertMemoryFormatToAclFormat(value_cache_tensor->Format());

  const bool is_key_cache_nz = (key_cache_format == ACL_FORMAT_FRACTAL_NZ);
  const bool is_value_cache_nz = (value_cache_format == ACL_FORMAT_FRACTAL_NZ);

  if (is_key_cache_nz && is_value_cache_nz) {
    param.kvCacheCfg = atb::infer::ReshapeAndCacheParam::K_CACHE_V_CACHE_NZ;
  } else {
    param.kvCacheCfg = atb::infer::ReshapeAndCacheParam::K_CACHE_V_CACHE;
  }

  // Real outputs are key_cache and value_cache; build a tuple Value from these inputs.
  std::vector<ir::ValuePtr> tuple_elems;
  tuple_elems.emplace_back(fxrt::ir::MakeIntrusive<ir::Value>(inputs[kKeyCacheIdx]->ToTensor()));
  tuple_elems.emplace_back(fxrt::ir::MakeIntrusive<ir::Value>(inputs[kValueCacheIdx]->ToTensor()));
  ir::TuplePtr out_tuple = fxrt::ir::MakeIntrusive<ir::Tuple>(std::move(tuple_elems));
  ir::ValuePtr out_value = fxrt::ir::MakeIntrusive<ir::Value>(out_tuple);

  auto& entry = GetOrCreateEntry(param, inputs, out_value.get());
  if (old_hash != current_hash_id_) {
    param_setter_.SetIndex({kKeyIdx, kValueIdx, kKeyCacheIdx, kValueCacheIdx, kSlotIndicesIdx}, {0, 1})
        .Input(inputs[kKeyIdx])
        .Input(inputs[kValueIdx])
        .Input(inputs[kKeyCacheIdx])
        .Input(inputs[kValueCacheIdx])
        .Input(inputs[kSlotIndicesIdx])
        .Output(inputs[kKeyCacheIdx])
        .Output(inputs[kValueCacheIdx]);
  }
  param_setter_.Update(inputs, out_value.get());
  return GetWorkspaceSize(entry, param_setter_.variant_pack, workspace_size);
}

OpsErrorCode AtbReshapeAndCache::Launch(
    const std::vector<const ir::Value*>& inputs,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  CHECK_IF_NULL(stream);
  return LaunchAtb(param_setter_.variant_pack, workspace, workspaceSize, static_cast<aclrtStream>(stream));
}

FXRT_REG_OP(reshape_and_cache, AtbReshapeAndCache, Ascend);

} // namespace ops
} // namespace fxrt
