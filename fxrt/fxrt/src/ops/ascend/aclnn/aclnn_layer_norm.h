#ifndef FXRT_OPS_ASCEND_ACLNN_ACLNN_LAYER_NORM_H_
#define FXRT_OPS_ASCEND_ACLNN_ACLNN_LAYER_NORM_H_

#include <memory>

#include "ops/ascend/aclnn/utils/aclnn_executor.h"
#include "ops/operator.h"

namespace fxrt {
namespace ops {

class AclnnLayerNorm : public Operator {
 public:
  AclnnLayerNorm();
  ~AclnnLayerNorm() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;

 private:
  std::unique_ptr<AclnnExecutor> executor_;
};

} // namespace ops
} // namespace fxrt

#endif // FXRT_OPS_ASCEND_ACLNN_ACLNN_LAYER_NORM_H_
