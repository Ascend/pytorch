#ifndef __OPS_OP_BASE_OP_REDUCE_SCATTER_H__
#define __OPS_OP_BASE_OP_REDUCE_SCATTER_H__

#include <utility>
#include <string>
#include <vector>

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class OpReduceScatter : public Operator {
 public:
  OpReduceScatter() = default;
  ~OpReduceScatter() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_OP_BASE_OP_REDUCE_SCATTER_H__
