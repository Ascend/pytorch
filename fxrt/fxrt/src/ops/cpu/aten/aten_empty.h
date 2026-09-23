#ifndef __OPS_CPU_ATEN_ATEN_EMPTY_H__
#define __OPS_CPU_ATEN_ATEN_EMPTY_H__

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class AtenEmpty : public Operator {
 public:
  AtenEmpty() = default;
  ~AtenEmpty() override = default;

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
#endif // __OPS_CPU_ATEN_ATEN_EMPTY_H__
