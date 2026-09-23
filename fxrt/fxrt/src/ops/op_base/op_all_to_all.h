#ifndef __OPS_OP_BASE_OP_ALL_TO_ALL_H__
#define __OPS_OP_BASE_OP_ALL_TO_ALL_H__

#include <utility>
#include <string>
#include <vector>

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class OpAllToAll : public Operator {
 public:
  OpAllToAll() = default;
  ~OpAllToAll() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_OP_BASE_OP_ALL_TO_ALL_H__
