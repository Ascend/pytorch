#ifndef __OPS_ASCEND_ACLNN_ACLNN_DYNAMIC_QUANT_V2_H__
#define __OPS_ASCEND_ACLNN_ACLNN_DYNAMIC_QUANT_V2_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
class AclnnDynamicQuantV2 : public Operator {
 public:
  AclnnDynamicQuantV2() {
    executor_ = std::make_unique<AclnnExecutor>("aclnnDynamicQuantV2");
  }
  ~AclnnDynamicQuantV2() override = default;

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
  int64_t outputType_{2};
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_DYNAMIC_QUANT_V2_H__
