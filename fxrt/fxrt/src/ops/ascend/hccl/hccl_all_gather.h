#ifndef OPS_ASCEND_HCCL_ALL_GATHER_H_
#define OPS_ASCEND_HCCL_ALL_GATHER_H_

#include <vector>

#include "ops/op_base/op_all_gather.h"

#include "ops/operator.h"
#include "ops/ascend/hccl/hccl_kernel.h"

namespace fxrt {
namespace ops {
class HcclAllGather : public OpAllGather {
 public:
  HcclAllGather() = default;
  ~HcclAllGather() = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;

 private:
  HcclKernel hcclKernel_;
};
} // namespace ops
} // namespace fxrt
#endif // OPS_ASCEND_HCCL_ALL_GATHER_H_
