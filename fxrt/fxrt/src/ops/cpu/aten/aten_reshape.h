#ifndef __OPS_CPU_ATEN_ATEN_RESHAPE_H__
#define __OPS_CPU_ATEN_ATEN_RESHAPE_H__

#include <vector>
#include <string>

#include "ops/op_base/op_reshape.h"

namespace fxrt {
namespace ops {
class AtenReshape : public Operator {
 public:
  AtenReshape() = default;
  ~AtenReshape() override = default;

  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
};
} // namespace ops
} // namespace fxrt

#endif // __OPS_CPU_ATEN_ATEN_RESHAPE_H__
