#ifndef __OPS_ASCEND_ACLNN_ACLNN_QUANT_MATMUL_H__
#define __OPS_ASCEND_ACLNN_ACLNN_QUANT_MATMUL_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {

class AclnnQuantMatmul : public Operator {
 public:
  AclnnQuantMatmul() {
    executor_ = std::make_unique<AclnnExecutor>("aclnnQuantMatmulV4");
    executorNz_ = std::make_unique<AclnnExecutor>("aclnnQuantMatmulWeightNz");
  }
  ~AclnnQuantMatmul() override = default;

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
  std::unique_ptr<AclnnExecutor> executorNz_{nullptr};
  ir::TensorPtr x1_{nullptr};
  ir::TensorPtr x2_{nullptr};
  ir::TensorPtr scale_{nullptr};
  ir::TensorPtr offset_{nullptr};
  ir::TensorPtr pertokenScale_{nullptr};
  ir::TensorPtr bias_{nullptr};
  ir::TensorPtr yscale_{nullptr};
  ir::TensorPtr x1Offset_{nullptr};
  ir::TensorPtr yOffset_{nullptr};
  bool isWeightNz_{false};
  bool transposeX1_{false};
  bool transposeX2_{false};
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_QUANT_MATMUL_H__
