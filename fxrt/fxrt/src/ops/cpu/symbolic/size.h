#ifndef __OPS_CPU_SYMBOLIC_SIZE_H__
#define __OPS_CPU_SYMBOLIC_SIZE_H__

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class Size : public Operator {
 public:
  Size() = default;
  ~Size() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
  bool NeedLaunch() override;
};

} // namespace ops
} // namespace fxrt
#endif // __OPS_CPU_SYMBOLIC_SIZE_H__
