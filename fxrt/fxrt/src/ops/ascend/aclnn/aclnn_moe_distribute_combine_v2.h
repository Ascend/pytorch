#ifndef __OPS_ASCEND_ACLNN_ACLNN_MOE_DISTRIBUTE_COMBINE_V2_H__
#define __OPS_ASCEND_ACLNN_ACLNN_MOE_DISTRIBUTE_COMBINE_V2_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
class AclnnMoeDistributeCombineV2 : public Operator {
 public:
  AclnnMoeDistributeCombineV2();
  ~AclnnMoeDistributeCombineV2() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;

 private:
  bool use_v4_{false};
  bool use_v3_{false};
  std::unique_ptr<AclnnExecutor> executor_v4_{nullptr};
  std::unique_ptr<AclnnExecutor> executor_v2_{nullptr};
  std::unique_ptr<AclnnExecutor> executor_v3_{nullptr};
  AclnnExecutor* active_executor_{nullptr};
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_MOE_DISTRIBUTE_COMBINE_V2_H__
