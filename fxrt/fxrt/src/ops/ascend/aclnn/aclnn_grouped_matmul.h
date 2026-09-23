#ifndef __OPS_ASCEND_ACLNN_ACLNN_GROUPED_MATMUL_H__
#define __OPS_ASCEND_ACLNN_ACLNN_GROUPED_MATMUL_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
constexpr int64_t DEFAULT_SPLIT = -1;

class AclnnGroupedMatmul : public Operator {
 public:
  AclnnGroupedMatmul() {
    executor_ = std::make_unique<AclnnExecutor>("aclnnGroupedMatmulV5");
    executorNz_ = std::make_unique<AclnnExecutor>("aclnnGroupedMatmulWeightNz");
  }
  ~AclnnGroupedMatmul() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;

 private:
  void ClearTensorList();
  std::unique_ptr<AclnnExecutor> executor_{nullptr};
  std::unique_ptr<AclnnExecutor> executorNz_{nullptr};
  ir::MemoryFormat weightFormat_;

  int64_t splitItem_{0};
  int64_t groupType_{DEFAULT_SPLIT};
  int64_t groupListType_{0};
  int64_t activateType_{0};
  int64_t quantPerGroupSize_{0};
  ir::TensorPtr groupList_{nullptr};
  std::vector<ir::TensorPtr> xList_;
  std::vector<ir::TensorPtr> weightList_;
  std::vector<ir::TensorPtr> biasList_;
  std::vector<ir::TensorPtr> scaleList_;
  std::vector<ir::TensorPtr> offsetList_;
  std::vector<ir::TensorPtr> antiquantScaleList_;
  std::vector<ir::TensorPtr> antiquantOffsetList_;
  std::vector<ir::TensorPtr> perTokenScaleList_;
  std::vector<ir::TensorPtr> activationInputList_;
  std::vector<ir::TensorPtr> activationQuantScaleList_;
  std::vector<ir::TensorPtr> activationQuantOffsetList_;
  std::vector<ir::TensorPtr> outputList_;
  std::vector<ir::TensorPtr> activationFeatureOutList_;
  std::vector<ir::TensorPtr> dynQuantScaleOutList_;
  std::vector<int64_t> tuningConfigList_;
};

} // namespace ops
} // namespace fxrt
#endif // #define __OPS_ASCEND_ACLNN_ACLNN_GROUPED_MATMUL_H__
