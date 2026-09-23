#ifndef __OPS_ASCEND_ACLNN_ACLNN_VIEW_BASE_H__
#define __OPS_ASCEND_ACLNN_ACLNN_VIEW_BASE_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"
#include "utils/view_utils.h"

namespace fxrt {
namespace ops {
class AclnnViewBase : public Operator {
 public:
  AclnnViewBase() = default;
  ~AclnnViewBase() override = default;

  void Init(const std::vector<const ir::Value*>& input, const ir::Value* output) override {
    refPairs_ = GenerateOutputInputRefPair(output);
  }

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override = 0;

  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override {
    return SUCCESS;
  }

  std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const override {
    return refPairs_;
  }

  bool NeedLaunch() override {
    return false;
  }

 protected:
  void CheckStorageMatch(const std::vector<const ir::Value*>& input, const ir::Value* output) {
    for (auto [outputIndex, inputIndex] : refPairs_) {
      auto& inputTensor = input[inputIndex]->ToTensor();

      // Get output tensor based on output type
      ir::TensorPtr outputTensor = nullptr;
      if (output->IsTensor()) {
        outputTensor = output->ToTensor();
      } else if (output->IsTuple()) {
        outputTensor = (*output->ToTuple())[outputIndex]->ToTensor();
      } else {
        RT_GLOG(EXCEPTION) << "Unsupported output type for ref input. outputIndex: " << outputIndex
                           << ", inputIndex: " << inputIndex << ", output: " << *output;
      }

      // Check if storage pointers match
      if (inputTensor->GetStorage()->Data() != outputTensor->GetStorage()->Data()) {
        RT_GLOG(EXCEPTION) << "Storage mismatch: Output tensor does not share the same storage pointer as input tensor."
                           << "outputIndex: " << outputIndex
                           << ", output storage: " << outputTensor->GetStorage()->Data()
                           << ", inputIndex: " << inputIndex
                           << ", input storage: " << inputTensor->GetStorage()->Data();
      }
    }
  }

 private:
  std::vector<std::pair<uint32_t, uint32_t>> refPairs_;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_VIEW_BASE_H__
