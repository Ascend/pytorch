#ifndef __OPS_OP_BASE_OP_RESHAPE_H__
#define __OPS_OP_BASE_OP_RESHAPE_H__

#include <utility>
#include <string>
#include <vector>

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class OpReshape : public Operator {
 public:
  OpReshape() = default;
  ~OpReshape() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
};
} // namespace ops
} // namespace fxrt
#endif // __OPS_OP_BASE_OP_RESHAPE_H__
