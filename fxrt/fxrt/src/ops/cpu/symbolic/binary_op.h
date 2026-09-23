#ifndef __OPS_CPU_SYMBOLIC_BINARY_OP_H__
#define __OPS_CPU_SYMBOLIC_BINARY_OP_H__

#include <vector>

#include "ops/operator.h"

namespace fxrt {
namespace ops {
class BinaryOp : public Operator {
 public:
  BinaryOp() = default;
  ~BinaryOp() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;
  OpsErrorCode Launch(
      const std::vector<const ir::Value*>& input,
      void* workspace,
      size_t workspaceSize,
      ir::Value* output,
      void* stream) override;
  bool NeedLaunch() override;
};

#define DefineBinaryOp(op_name)     \
  class op_name : public BinaryOp { \
   public:                          \
    op_name() = default;            \
    ~op_name() override = default;  \
  }

DefineBinaryOp(BinaryAdd);
DefineBinaryOp(BinarySub);
DefineBinaryOp(BinaryMul);
DefineBinaryOp(BinaryDiv);
DefineBinaryOp(BinaryFloorDiv);
} // namespace ops
} // namespace fxrt

#endif // __OPS_CPU_SYMBOLIC_BINARY_OP_H__
