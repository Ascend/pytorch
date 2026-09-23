#ifndef __OPS_OP_BASE_OP_EVAL_SYMBOLIC_EXPR_H__
#define __OPS_OP_BASE_OP_EVAL_SYMBOLIC_EXPR_H__

#include "ops/operator.h"

namespace fxrt {
namespace ops {

class OpEvalSymbolicExpr : public Operator {
 public:
  OpEvalSymbolicExpr() = default;
  ~OpEvalSymbolicExpr() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) override;

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

#endif // __OPS_OP_BASE_OP_EVAL_SYMBOLIC_EXPR_H__
