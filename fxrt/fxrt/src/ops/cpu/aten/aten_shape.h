#ifndef __OPS_CPU_ATEN_ATEN_SHAPE_H__
#define __OPS_CPU_ATEN_ATEN_SHAPE_H__

#include <vector>
#include <string>

#include "ops/op_base/op_shape.h"

namespace fxrt {
namespace ops {
class AtenShape : public OpShape {
 public:
  AtenShape() = default;
  ~AtenShape() override = default;

  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;

  bool NeedLaunch() override;
};
} // namespace ops
} // namespace fxrt

#endif // __OPS_CPU_ATEN_ATEN_SHAPE_H__
