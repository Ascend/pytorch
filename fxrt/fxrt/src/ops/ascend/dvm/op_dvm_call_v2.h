#ifndef __OPS_ASCEND_DVM_OP_DVM_CALL_V2_H__
#define __OPS_ASCEND_DVM_OP_DVM_CALL_V2_H__

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <vector>

#include "ir/tensor/tensor.h"
#include "ops/op_register.h"

namespace dvm {
class Kernel;
class NDObject;
struct RelocEntry;
} // namespace dvm

namespace fxrt {
namespace ops {

class AclnnExecutor;

class OpDvmCallV2 : public Operator {
 public:
  OpDvmCallV2() = default;
  ~OpDvmCallV2() override;

  void Init(const std::vector<const ir::Value*>& inputs, const ir::Value* output) override;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;

  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;

 private:
  struct ContiguousCopyPlan {
    ContiguousCopyPlan();
    ~ContiguousCopyPlan();
    ContiguousCopyPlan(ContiguousCopyPlan&&) noexcept;
    ContiguousCopyPlan& operator=(ContiguousCopyPlan&&) noexcept;
    ContiguousCopyPlan(const ContiguousCopyPlan&) = delete;
    ContiguousCopyPlan& operator=(const ContiguousCopyPlan&) = delete;

    size_t realInputIndex{0};
    size_t inputIndex{0};
    size_t copyWorkspaceOffset{0};
    size_t copyWorkspaceSize{0};
    size_t tempBufferOffset{0};
    size_t tempBufferSize{0};
    ir::TensorPtr tempTensor;
  };

  void ResolveKernelObject(const std::string& handle);
  void RefreshKernelState(const pybind11::object& kernelObj);
  void UpdateOutputMetadata(ir::Value* output) const;
  std::vector<void*> BuildAddressVector(
      const std::vector<const ir::Value*>& inputs,
      ir::Value* output,
      const std::vector<void*>* inputAddrOverrides,
      bool allowNonContiguousInputs) const;
  size_t PlanContiguousInputs(size_t offset);
  std::vector<void*> PrepareContiguousInputs(void* workspace, size_t workspaceSize, void* stream);
  void UpdateDynamicShapeRefs();

  std::string handle_;
  dvm::Kernel* rawKernel_{nullptr};
  std::vector<dvm::RelocEntry>* relocs_{nullptr};
  std::vector<dvm::NDObject*>* loads_{nullptr};
  std::vector<dvm::NDObject*>* stores_{nullptr};
  size_t numTensorInputs_{0};
  size_t numOutputs_{0};
  size_t workspaceSize_{0};
  size_t totalWorkspaceSize_{0};
  bool isDynamic_{false};
  bool isSplit_{false};
  std::vector<const ir::Value*> realInputs_;
  std::vector<std::vector<int64_t>> dynamicInputShapes_;
  std::vector<std::vector<int64_t>> dynamicInputStrides_;
  std::vector<ContiguousCopyPlan> contiguousCopyPlans_;
  std::unique_ptr<pybind11::object> kernelObj_;
};

} // namespace ops
} // namespace fxrt

#endif // __OPS_ASCEND_DVM_OP_DVM_CALL_V2_H__
