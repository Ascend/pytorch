#ifndef __OPS_ASCEND_MEM_EMPTY_STRIDED_H__
#define __OPS_ASCEND_MEM_EMPTY_STRIDED_H__

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class EmptyStrided : public Operator {
 public:
  EmptyStrided() = default;
  ~EmptyStrided() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value*>& input, const ir::Value* output, size_t* workspaceSize)
      override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
  bool NeedLaunch() override {
    return false;
  }
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_MEM_EMPTY_STRIDED_H__
