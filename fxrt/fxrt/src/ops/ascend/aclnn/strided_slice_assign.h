#ifndef __OPS_ASCEND_ACLNN_ACLNN_STRIDED_SLICE_ASSIGN_H__
#define __OPS_ASCEND_ACLNN_ACLNN_STRIDED_SLICE_ASSIGN_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {

class AclnnStridedSliceAssign : public Operator {
 public:
  AclnnStridedSliceAssign() {
    executor_ = std::make_unique<AclnnExecutor>("aclnnStridedSliceAssignV2");
  }
  ~AclnnStridedSliceAssign() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;

 private:
  std::unique_ptr<AclnnExecutor> executor_{nullptr};
  ir::TensorPtr x_;
  ir::TensorPtr value_;
  std::vector<int64_t> begin_;
  std::vector<int64_t> end_;
  std::vector<int64_t> strides_;
  std::optional<std::vector<int64_t>> axes_;
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_STRIDED_SLICE_ASSIGN_H__
