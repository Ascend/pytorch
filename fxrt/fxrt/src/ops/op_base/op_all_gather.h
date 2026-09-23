#ifndef __OPS_OP_BASE_OP_ALL_GATHER_H__
#define __OPS_OP_BASE_OP_ALL_GATHER_H__

#include <utility>
#include <vector>
#include <string>

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class OpAllGather : public Operator {
 public:
  OpAllGather() = default;
  ~OpAllGather() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_OP_BASE_OP_ALL_GATHER_H__
