#ifndef __OPS_ASCEND_MEM_EMPTY_H__
#define __OPS_ASCEND_MEM_EMPTY_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace fxrt {
namespace ops {
class Empty : public Operator {
 public:
  Empty() = default;
  ~Empty() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_MEM_EMPTY_H__
