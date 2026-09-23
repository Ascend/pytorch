#include "ops/op_base/op_eval_symbolic_expr.h"

#include "ir/value/value.h"
#include "ir/symbolic/symbolic.h"
#include "common/logger.h"

namespace fxrt {
namespace ops {

// Inputs: [operand_0, operand_1, ..., operand_N-1, symVars]
// Output: symExpr
// Set the value of symVars with operands, such that symExpr in output can be evaluated later.
OpsErrorCode OpEvalSymbolicExpr::InferShape(const std::vector<const ir::Value*>& input, ir::Value* output) {
  if (input.size() < 1) {
    RT_GLOG(EXCEPTION) << "OpEvalSymbolicExpr: input size must be at least 1";
  }

  size_t numOperands = input.size() - 1;
  auto symVars = input[numOperands]->ToTuple();

  if (symVars->Size() != numOperands) {
    RT_GLOG(EXCEPTION) << "OpEvalSymbolicExpr: symVars size must be equal to numOperands";
  }

  for (size_t i = 0; i < numOperands; ++i) {
    auto symVarExpr = (*symVars)[i]->ToSymbol();
    auto symVar = static_cast<ir::SymbolicVar*>(symVarExpr.get());
    symVar->SetValue(input[i]->ToInt());
  }

  return SUCCESS;
}

OpsErrorCode OpEvalSymbolicExpr::Launch(
    const std::vector<const ir::Value*>& input,
    void* workspace,
    size_t workspaceSize,
    ir::Value* output,
    void* stream) {
  return SUCCESS;
}

bool OpEvalSymbolicExpr::NeedLaunch() {
  return false;
}
} // namespace ops
} // namespace fxrt
