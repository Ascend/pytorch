#ifndef OPS_ASCEND_HCCL_WAIT_TENSOR_H_
#define OPS_ASCEND_HCCL_WAIT_TENSOR_H_

#include <vector>

#include "ops/operator.h"
#include "ops/ascend/hccl/hccl_kernel.h"

namespace fxrt {
namespace ops {
class HcclWaitTensor : public Operator {
 public:
  HcclWaitTensor() = default;
  ~HcclWaitTensor() = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
  bool NeedLaunch() override;
  std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const override {
    return {std::pair<uint32_t, uint32_t>(0, 0)};
  }
};
} // namespace ops
} // namespace fxrt
#endif // OPS_ASCEND_HCCL_WAIT_TENSOR_H_
