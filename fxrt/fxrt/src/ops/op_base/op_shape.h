#ifndef __OPS_OP_BASE_OP_SHAPE_H__
#define __OPS_OP_BASE_OP_SHAPE_H__

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class OpShape : public Operator {
 public:
  OpShape() = default;
  ~OpShape() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_OP_BASE_OP_SHAPE_H__
