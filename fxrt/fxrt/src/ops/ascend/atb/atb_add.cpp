#include "ops/ascend/atb/atb_add.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

AtbAdd::AtbAdd() : AtbBase("add") {}

OpsErrorCode AtbAdd::CalcWorkspace(
    const std::vector<const ir::Value*>& inputs,
    const ir::Value* output,
    size_t* workspace_size) {
  CHECK_IF_NULL(workspace_size);
  if (inputs.size() < 2) {
    RT_GLOG(ERROR) << "Invalid parameters for AtbAdd::CalcWorkspace, input size: " << inputs.size();
    return OpsErrorCode::INVALID_PARAM;
  }
  auto old_hash = current_hash_id_;
  atb::infer::ElewiseParam param;
  param.elewiseType = atb::infer::ElewiseParam::ELEWISE_ADD;
  // Update hash id in this func
  auto& entry = GetOrCreateEntry(param, inputs, output);
  if (old_hash != current_hash_id_) {
    param_setter_.SetIndex({0, 1}, {0}).Input(inputs[0]).Input(inputs[1]).Output(output);
  }
  param_setter_.Update(inputs, output);
  return GetWorkspaceSize(entry, param_setter_.variant_pack, workspace_size);
}

OpsErrorCode AtbAdd::Launch(
    const std::vector<const ir::Value*>& inputs,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  CHECK_IF_NULL(stream);
  return LaunchAtb(param_setter_.variant_pack, workspace, workspaceSize, static_cast<aclrtStream>(stream));
}

} // namespace ops
} // namespace fxrt
