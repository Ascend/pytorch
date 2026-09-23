#include <vector>

#include "ops/op_base/op_shape.h"
#include "ir/value/value.h"
#include "common/logger.h"

namespace fxrt {
namespace ops {
OpsErrorCode OpShape::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  // The output of Op Shape is a tuple, which does not have a shape in the tensor sense.
  auto tensor = input[kIndex0]->ToTensor();
  auto& shape = tensor->Shape();

  std::vector<ir::ValuePtr> shapeValues;
  shapeValues.reserve(shape.size());
  for (auto dim : shape) {
    (void)shapeValues.emplace_back(ir::MakeIntrusive<ir::Value>(dim));
  }

  auto tuple_data = ir::MakeIntrusive<ir::Tuple>(shapeValues);
  *output = ir::Value(tuple_data);

  return SUCCESS;
}
} // namespace ops
} // namespace fxrt
