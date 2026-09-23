#ifndef __OPS_TORCH_OP_TORCH_CALL_H__
#define __OPS_TORCH_OP_TORCH_CALL_H__

#include <torch/torch.h>
#include <string>
#include <vector>
#include "ops/op_register.h"
#include "ir/value/value.h"

namespace fxrt {
namespace ops {
class OpTorchCall : public Operator {
 public:
  explicit OpTorchCall(const std::string& opName) {
    qualifiedOpName_ = opName;
    auto pos = qualifiedOpName_.find(".");
    if (pos != std::string::npos) {
      qualifiedOpName_.replace(pos, 1, "::");
    }
  }
  ~OpTorchCall() override = default;

  void Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output);

  OpsErrorCode CalcWorkspace(
      const std::vector<const ir::Value*>& input,
      const ir::Value* output,
      size_t* workspaceSize);
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream);
  bool NeedLaunch() override;

 protected:
  void ConvertInputsToStack(const std::vector<const ir::Value*>& inputs, torch::jit::Stack& stack);
  void ConvertStackToOutput(ir::Value* output, torch::jit::Stack&& stack) const;
  void ToFxrtTensor(ir::Value* output, torch::jit::IValue&& ivalue) const;
  bool MatchOpSchema(
      const std::vector<const ir::Value*>& inputs,
      const std::shared_ptr<torch::jit::Operator> op,
      std::string* mismatch_reason = nullptr) const;
  bool HasSharedStorageWithInput(const ir::Value* output, const ir::Value* input) const;
  std::string GetInputTypesExpr(const std::vector<const ir::Value*>& inputs) const;
  std::string GetAvailableTorchOps() const;

  void ConvertTensorInputToStack(const ir::Value* value, torch::jit::Stack& stack);
  void ConvertDoubleInputToStack(const ir::Value* value, torch::jit::Stack& stack);
  void ConvertIntInputToStack(const ir::Value* value, torch::jit::Stack& stack);
  void ConvertBoolInputToStack(const ir::Value* value, torch::jit::Stack& stack);
  void ConvertStringInputToStack(const ir::Value* value, torch::jit::Stack& stack);
  void ConvertTupleInputToStack(const ir::Value* value, torch::jit::Stack& stack);
  void ConvertNoneInputToStack(const ir::Value* value, torch::jit::Stack& stack);

  void ConvertTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack);
  void ConvertTensorTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack);
  void ConvertIntTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack);
  void ConvertBoolTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack);
  void ConvertDoubleTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack);
  void ConvertStringTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack);
  void ConvertNoneTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack& stack);
  void UpdateTorchTensor(at::Tensor& atTensor, const ir::TensorPtr& fxrtTensor);

  using ConvertInputsFunc = void (OpTorchCall::*)(const ir::Value*, torch::jit::Stack&);
  using ConvertTupleFunc = void (OpTorchCall::*)(const ir::TuplePtr, torch::jit::Stack&);

  // Jump tables for O(1) type dispatch
  static const ConvertInputsFunc inputConverterTable[];
  static const ConvertTupleFunc tupleConverterTable[];
  static constexpr size_t kInputConverterCount = 8;
  static constexpr size_t kTupleConverterCount = 8;

  std::string qualifiedOpName_;
  torch::jit::Operation operation_ = nullptr;
  std::vector<at::Tensor> atTensors_;
  size_t tensorIdx_ = 0;
  bool firstRun_ = true;

  // Cache input converters to avoid runtime tag lookup
  std::vector<ConvertInputsFunc> cachedInputConverters_;
};
} // namespace ops
} // namespace fxrt

#endif // __OPS_TORCH_OP_TORCH_CALL_H__
