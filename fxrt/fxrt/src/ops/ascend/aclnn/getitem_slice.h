#ifndef __OPS_ASCEND_ACLNN_ACLNN_GETITEM_SLICE_H__
#define __OPS_ASCEND_ACLNN_ACLNN_GETITEM_SLICE_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
class AclnnGetItemSlice : public Operator {
 public:
  AclnnGetItemSlice() {
    executor_ = std::make_unique<AclnnExecutor>("aclnnSliceV2");
  }
  ~AclnnGetItemSlice() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
  bool NeedLaunch() override;

 private:
  std::unique_ptr<AclnnExecutor> executor_{nullptr};
  ir::TensorPtr dst_;
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> axes_;
  std::vector<int64_t> steps_;
  bool needSqueeze_{false};
  bool skipLaunch_{false};
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_GETITEM_SLICE_H__
