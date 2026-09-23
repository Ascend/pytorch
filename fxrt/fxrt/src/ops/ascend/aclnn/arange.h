#ifndef __OPS_ASCEND_ACLNN_ACLNN_ARANGE_H__
#define __OPS_ASCEND_ACLNN_ACLNN_ARANGE_H__

#include <vector>
#include <memory>

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
class AclnnArange : public Operator {
 public:
  AclnnArange() {
    executor_ = std::make_unique<AclnnExecutor>("aclnnArange");
  }
  ~AclnnArange() override = default;

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
#endif // __OPS_ASCEND_ACLNN_ACLNN_ARANGE_H__
