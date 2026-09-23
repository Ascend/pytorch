#ifndef OPS_ASCEND_HCCL_ALL_TO_ALL_H_
#define OPS_ASCEND_HCCL_ALL_TO_ALL_H_
#include <vector>

#include "ops/op_base/op_all_to_all.h"

#include "ops/operator.h"
#include "ops/ascend/hccl/hccl_kernel.h"

namespace fxrt {
namespace ops {
class HcclAllToAll : public OpAllToAll {
 public:
  HcclAllToAll() = default;
  ~HcclAllToAll() = default;

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
  bool useAllToAllV_;
};
} // namespace ops
} // namespace fxrt
#endif // OPS_ASCEND_HCCL_ALL_TO_ALL_H_
