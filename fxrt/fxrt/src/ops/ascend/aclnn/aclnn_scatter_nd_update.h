#ifndef __OPS_ASCEND_ACLNN_ACLNN_SCATTER_ND_UPDATE_H__
#define __OPS_ASCEND_ACLNN_ACLNN_SCATTER_ND_UPDATE_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
class AclnnScatterNdUpdateNonInplace : public Operator {
 public:
  AclnnScatterNdUpdateNonInplace() {
    executor_ = std::make_unique<AclnnExecutor>("aclnnScatterNdUpdate");
  }
  ~AclnnScatterNdUpdateNonInplace() override = default;

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
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_SCATTER_ND_UPDATE_H__
