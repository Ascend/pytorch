#ifndef __OPS_OP_BASE_OP_MATMUL_H__
#define __OPS_OP_BASE_OP_MATMUL_H__

#include <utility>
#include <string>

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class OpMatMul : public Operator {
 public:
  OpMatMul() = default;
  ~OpMatMul() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_OP_BASE_OP_MATMUL_H__
