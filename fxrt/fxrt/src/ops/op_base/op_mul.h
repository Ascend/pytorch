#ifndef __OPS_OP_BASE_OP_MUL_H__
#define __OPS_OP_BASE_OP_MUL_H__

#include <vector>

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class OpMul : public Operator {
 public:
  OpMul() = default;
  ~OpMul() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_OP_BASE_OP_MUL_H__
