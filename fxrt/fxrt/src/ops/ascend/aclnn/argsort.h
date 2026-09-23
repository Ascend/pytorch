#ifndef __OPS_ASCEND_ACLNN_ARGSORT_H__
#define __OPS_ASCEND_ACLNN_ARGSORT_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
class AclnnArgsort : public Operator {
 public:
  AclnnArgsort() {
    executor_ = std::make_unique<AclnnExecutor>("aclnnSort");
  }
  ~AclnnArgsort() override = default;

  void Init(const std::vector<const ir::Value*>& input, const ir::Value* output) override;
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

#endif // __OPS_ASCEND_ACLNN_ARGSORT_H__
