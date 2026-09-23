#ifndef __OPS_CPU_SYMBOLIC_BROADCAST_SHAPE_H__
#define __OPS_CPU_SYMBOLIC_BROADCAST_SHAPE_H__

#include "ops/operator.h"

namespace fxrt {
namespace ops {

// Compute broadcasted shape of two tensors and return it as a tuple<int64>.
// This is a host-side metadata op used by dynamic-shape lowering.
class BroadcastShape : public Operator {
 public:
  BroadcastShape() = default;
  ~BroadcastShape() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
};

} // namespace ops
} // namespace fxrt

#endif // __OPS_CPU_SYMBOLIC_BROADCAST_SHAPE_H__
