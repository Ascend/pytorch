#ifndef OPS_ASCEND_HCCL_ALL_REDUCE_H_
#define OPS_ASCEND_HCCL_ALL_REDUCE_H_

#include <vector>

#include "ops/op_base/op_all_reduce.h"

#include "ops/operator.h"
#include "ops/ascend/hccl/hccl_kernel.h"

namespace fxrt {
namespace ops {
class HcclAllReduce : public OpAllReduce {
 public:
  HcclAllReduce() = default;
  ~HcclAllReduce() = default;

  void Init(const std::vector<const ir::Value*>& input, const ir::Value* output) override;
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
  HcclReduceOp hcclOpType_{HCCL_REDUCE_SUM};
};
} // namespace ops
} // namespace fxrt
#endif // OPS_ASCEND_HCCL_ALL_REDUCE_H_
