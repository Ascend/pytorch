#include "ops/ascend/atb/atb_linear.h"
#include "ops/op_register.h"

namespace fxrt {
namespace ops {

OpsErrorCode AtbLinear::CalcWorkspace(
    const std::vector<const ir::Value*>& inputs,
    const ir::Value* output,
    size_t* workspace_size) {
  CHECK_IF_NULL(workspace_size);
  if (inputs.size() < 2) {
    RT_GLOG(ERROR) << "Invalid parameters for AtbLinear::CalcWorkspace, input size: " << inputs.size();
    return OpsErrorCode::INVALID_PARAM;
  }

  auto old_hash = current_hash_id_;
  atb::infer::LinearParam param;
  param.hasBias = !IsBiasNone(inputs);
  param.transposeA = false; // Input tensor usually does not need transpose
  param.transposeB = true; // Weight tensor is usually transposed
  param.enAccum = false; // Accumulation is not needed

  // Update hash id in this func
  auto& entry = GetOrCreateEntry(param, inputs, output);
  if (old_hash != current_hash_id_) {
    if (param.hasBias) {
      param_setter_.SetIndex({0, 1, 2}, {0}).Input(inputs[0]).Input(inputs[1]).Input(inputs[2]).Output(output);
    } else {
      param_setter_.SetIndex({0, 1}, {0}).Input(inputs[0]).Input(inputs[1]).Output(output);
    }
  }
  param_setter_.Update(inputs, output);
  return GetWorkspaceSize(entry, param_setter_.variant_pack, workspace_size);
}

OpsErrorCode AtbLinear::Launch(
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
