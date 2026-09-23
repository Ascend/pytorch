#ifndef __OPS_OP_BASE_OP_ALL_REDUCE_H__
#define __OPS_OP_BASE_OP_ALL_REDUCE_H__

#include <utility>
#include <string>
#include <vector>

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class OpAllReduce : public Operator {
 public:
  OpAllReduce() = default;
  ~OpAllReduce() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_OP_BASE_OP_ALL_REDUCE_H__
