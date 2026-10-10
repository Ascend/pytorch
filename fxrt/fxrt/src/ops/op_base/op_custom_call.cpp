#include "ops/op_base/op_custom_call.h"
#include "ops/custom_op_register.h"
#ifdef ENABLE_TORCH_FRONT
#include "ops/op_base/op_torch_call.h"
#endif

namespace fxrt {
namespace ops {
constexpr size_t kInputIOpNameIndex = 0;
constexpr size_t kRealInputIndex = 1;

void OpCustomCall::Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output) {
  CHECK_IF_NULL(inputs[kInputIOpNameIndex]);
  opName_ = inputs[kInputIOpNameIndex]->ToString();
  size_t pos = opName_.find(".");
  if (pos == std::string::npos) {
    RT_GLOG(EXCEPTION) << "Invalid op name: " << opName_ << ". Op name must be in the format of ns.op_name.";
  }
  std::string opName = opName_.substr(pos + 1);
  operatorPtr_ = CreateCustomOperator(opName);
  SetOpType(OpType::CustomCallOp);
#ifdef ENABLE_TORCH_FRONT
  if (operatorPtr_ == nullptr) {
    RT_VLOG(VL_OPS) << "Custom op " << opName_ << " not registered. Try to create operator from torch.";
    operatorPtr_ = std::make_shared<OpTorchCall>(opName_);
    SetOpType(OpType::TorchCallOp);
  }
#endif
  CHECK_IF_NULL(operatorPtr_);
  operatorPtr_->Init(inputs, output);
  auto inputSize = inputs.size() - kRealInputIndex;
  input_.resize(inputSize, nullptr);
  for (size_t i = kRealInputIndex; i < inputs.size(); i++) {
    input_[i - kRealInputIndex] = inputs[i];
  }

  // The input indices must be translated: the inner operator numbers its inputs from the first real
  // argument, while the runtime (OpRunner::input_) still sees the custom-call node inputs, whose
  // element 0 is the operator name. So every inner input index is shifted by the op name offset.
  refPairs_.clear();
  refPairs_ = operatorPtr_->GetOutputInputRefPairs();
  for (auto& refPair : refPairs_) {
    refPair.second += kRealInputIndex;
    RT_VLOG(VL_OPS) << "OpCustomCall: " << opName_ << " ref pair: output[" << refPair.first << "] aliases input["
                    << refPair.second << "]";
  }
}

OpsErrorCode OpCustomCall::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  if (operatorPtr_ == nullptr) {
    RT_GLOG(ERROR) << "operatorPtr_ is null in OpCustomCall::InferShape";
    return UNKNOWN_ERROR;
  }
  return operatorPtr_->InferShape(input_, output);
}

OpsErrorCode OpCustomCall::CalcWorkspace(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    size_t* workspaceSize) {
  return operatorPtr_->CalcWorkspace(input_, output, workspaceSize);
}

OpsErrorCode OpCustomCall::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  return operatorPtr_->Launch(input_, workspace, workspaceSize, output, stream);
}

} // namespace ops
} // namespace fxrt
