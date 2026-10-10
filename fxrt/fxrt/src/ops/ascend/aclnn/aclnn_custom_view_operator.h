#ifndef __OPS_ASCEND_ACLNN_ACLNN_CUSTOM_VIEW_OPERATOR_H__
#define __OPS_ASCEND_ACLNN_ACLNN_CUSTOM_VIEW_OPERATOR_H__

#include <utility>
#include <vector>

#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/operator.h"

namespace fxrt {
namespace ops {

/**
 * @brief Base class for zero-copy view (ref) custom operators.
 *
 * A view custom operator produces an output that shares the storage of its input and only reinterprets
 * the data through different metadata (shape / strides / storageOffset). No kernel is launched.
 *
 * Difference from AclnnCustomOperator: the output shape usually differs from the input shape, so the
 * generic ref metadata sync in OpRunner::UpdateRefNodeOutputMetadata() deliberately skips these
 * operators. The subclass is therefore responsible for computing the view metadata in CalcWorkspace()
 * and applying it via UpdateTensorViewInfo(), exactly like the built-in view operators do.
 *
 * Usage:
 * @code
 * class MyCustomSliceOperator : public AclnnCustomViewOperator {
 *  public:
 *   OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
 *                              size_t *workspaceSize) override {
 *     const auto inputTensor = input[kIndex0]->ToTensor();
 *     const auto outputTensor = output->ToTensor();
 *     auto newStrides = GetTensorStrides(inputTensor);
 *     size_t newStorageOffset = ...;
 *     UpdateTensorViewInfo(inputTensor, outputTensor, outputTensor->Shape(), newStrides, newStorageOffset);
 *     CheckStorageMatch(input, output);
 *     return SUCCESS;
 *   }
 * };
 * REGISTER_CUSTOM_OP(my_custom_slice, MyCustomSliceOperator);
 * @endcode
 *
 * Helpers available from "ops/ascend/aclnn/utils/view_utils.h": CalculateViewStrides, CalculateStrides,
 * GetTensorStrides, DynamicDimWrap, UpdateTensorViewInfo.
 */
class DA_API AclnnCustomViewOperator : public Operator {
 public:
  AclnnCustomViewOperator() = default;
  ~AclnnCustomViewOperator() override = default;

  /**
   * @brief Declare the output-input aliasing.
   * A tensor output produces {{0, 0}}; a tuple output produces {{i, 0}} for every element, i.e. all
   * outputs alias input 0.
   */
  void Init(const std::vector<const ir::Value*>& input, const ir::Value* output) override {
    refPairs_ = GenerateOutputInputRefPair(output);
  }

  /**
   * @brief Compute and apply the view metadata for the output(s).
   * Subclasses must update the output tensor metadata here, typically with UpdateTensorViewInfo().
   */
  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override = 0;

  // A view operator only manipulates metadata, so there is no kernel to launch.
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override {
    return SUCCESS;
  }

  bool NeedLaunch() override {
    return false;
  }

  std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const override {
    return refPairs_;
  }

 protected:
  /**
   * @brief Verify that every declared ref output really shares the input storage.
   * Call this at the end of CalcWorkspace() to catch a subclass that forgot to keep the alias.
   *
   * Note: the diagnostics here deliberately avoid streaming ir::Value, because
   * operator<<(std::ostream &, const ir::Value &) is not exported from the runtime libraries and
   * would leave an undefined symbol in a separately compiled custom operator .so.
   */
  void CheckStorageMatch(const std::vector<const ir::Value*>& input, const ir::Value* output) const {
    for (auto [outputIndex, inputIndex] : refPairs_) {
      CHECK_IF_FAIL(inputIndex < input.size());
      CHECK_IF_NULL(input[inputIndex]);
      const auto& inputTensor = input[inputIndex]->ToTensor();

      ir::TensorPtr outputTensor = nullptr;
      if (output->IsTensor()) {
        outputTensor = output->ToTensor();
      } else if (output->IsTuple()) {
        const auto& outputTuple = output->ToTuple();
        CHECK_IF_FAIL(outputIndex < outputTuple->Size());
        outputTensor = (*outputTuple)[outputIndex]->ToTensor();
      } else {
        RT_GLOG(EXCEPTION) << "Custom view operator: unsupported output type for ref input, it must be a tensor "
                           << "or a tuple. outputIndex: " << outputIndex << ", inputIndex: " << inputIndex;
      }

      if (inputTensor->GetStorage()->Data() != outputTensor->GetStorage()->Data()) {
        RT_GLOG(EXCEPTION) << "Custom view operator: output tensor does not share the same storage pointer as the "
                           << "input tensor. outputIndex: " << outputIndex
                           << ", output storage: " << outputTensor->GetStorage()->Data()
                           << ", inputIndex: " << inputIndex
                           << ", input storage: " << inputTensor->GetStorage()->Data();
      }
    }
  }

  /**
   * @brief Update the declared ref pairs when the default (all outputs alias input 0) does not apply.
   */
  void SetOutputInputRefPairs(std::vector<std::pair<uint32_t, uint32_t>> refPairs) {
    refPairs_ = std::move(refPairs);
  }

 private:
  std::vector<std::pair<uint32_t, uint32_t>> refPairs_;
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_CUSTOM_VIEW_OPERATOR_H__
