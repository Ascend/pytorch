#ifndef OPS_ASCEND_HCCL_TENSOR_COPY_H_
#define OPS_ASCEND_HCCL_TENSOR_COPY_H_

#include <vector>

#include "ops/operator.h"
#include "ops/ascend/hccl/hccl_kernel.h"

namespace fxrt {
namespace ops {
class HcclTensorCopy : public Operator {
 public:
  HcclTensorCopy() = default;
  ~HcclTensorCopy() = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;

  OpsErrorCode CalcWorkspace(
      const std::vector<const ir::Value*>& input,
      const ir::Value* output,
      size_t* workspace_size) override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
};
} // namespace ops
} // namespace fxrt
#endif // OPS_ASCEND_HCCL_TENSOR_COPY_H_
