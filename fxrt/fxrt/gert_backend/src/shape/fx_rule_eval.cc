#include "shape/fx_rule_eval.h"

#include <exception>

#include "common/logger.h"

namespace fx_converter {
namespace {

// fxrt::ir::SymbolicExpr::Evaluate() -- plain int64 arithmetic against the values BindSymbolEnv
// wrote straight into the SymbolicVar objects.
//
// Evaluate() reports division-by-zero and unbound variables through RT_GLOG(EXCEPTION), i.e. it
// throws. The GE kernel boundary is a C-style return code, so the throw must be contained here.
bool EvalDimsNative(const FxShapeRule &rule, std::vector<int64_t> &dims) {
  if (rule.out_dims_ir.size() != dims.size()) {
    return false;
  }
  try {
    for (size_t i = 0; i < dims.size(); ++i) {
      if (rule.out_dims_ir[i] == nullptr) {
        return false;
      }
      dims[i] = rule.out_dims_ir[i]->Evaluate();
      // Zero is a legitimate dim (an empty tensor axis); only a negative result is broken.
      if (dims[i] < 0) {
        return false;
      }
    }
  } catch (const std::exception &e) {
    RT_GLOG(WARNING) << "EvalRuleDims: native Evaluate threw (" << e.what() << "), source=" << rule.source;
    return false;
  }
  return true;
}

}  // namespace

bool RuleIsEvaluable(const FxShapeRule *rule) {
  if (rule == nullptr) {
    return false;
  }
  if (rule->out_dims_ir.empty()) {
    // A rank-0 output has no dims to evaluate and needs no symbol binding.
    return rule->is_rank0;
  }
  if (rule->env == nullptr || !rule->env->bound) {
    RT_GLOG(ERROR) << "EvalRuleDims: SymbolEnv not bound (source=" << rule->source << ")";
    return false;
  }
  return true;
}

bool EvalRuleDims(const FxShapeRule &rule, std::vector<int64_t> &dims) {
  const size_t rank = rule.out_dims_ir.size();
  dims.assign(rank, 0);
  if (rank == 0U) {
    // Only a deliberately marked rank-0 rule yields an empty dim list successfully.
    return rule.is_rank0;
  }
  return EvalDimsNative(rule, dims);
}

}  // namespace fx_converter
