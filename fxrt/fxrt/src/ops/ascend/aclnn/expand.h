#ifndef __OPS_ASCEND_ACLNN_ACLNN_EXPAND_H__
#define __OPS_ASCEND_ACLNN_ACLNN_EXPAND_H__

#include <vector>
#include <memory>

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
class AclnnExpand : public Operator {
 public:
  AclnnExpand();
  ~AclnnExpand() override = default;

  // NOTE: `fxrt.expand` output shape may contain -1 (dynamic) in IR types.
  // In dynamic-shape execution, runtime calls InferShape() before CalcWorkspace()/Launch.
  // We must resolve the real output shape here (per expand semantics) to avoid
  // "Tensor shape still unknown before launch".
  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;

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
  std::vector<int64_t> resolved_size_; // resolved size from InferShape (with -1 replaced)
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_ACLNN_EXPAND_H__
