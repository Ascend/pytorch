#ifndef __OPS_ASCEND_ACLNN_ACLNN_CONTIGUOUS_H__
#define __OPS_ASCEND_ACLNN_ACLNN_CONTIGUOUS_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
class AclnnContiguous : public Operator {
 public:
  AclnnContiguous() {
    executor_ = std::make_unique<AclnnExecutor>("aclnnInplaceCopy");
  }
  ~AclnnContiguous() override = default;

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
  bool srcContiguous_{false};
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_CONTIGUOUS_H__
