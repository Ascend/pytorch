#pragma once

#include <cstdint>
#include <vector>

#include "shape/fx_shape_rule.h"

namespace fx_converter {

// Evaluate one rule's output dims for the current Execute with fxrt::ir::SymbolicExpr::Evaluate().
// Resizes `dims` to the rule's rank. Shared by the per-op InferShape kernel and the fused one.
bool EvalRuleDims(const FxShapeRule& rule, std::vector<int64_t>& dims);

// True when the rule is usable: bound env, or a rank-0 rule with no dims.
bool RuleIsEvaluable(const FxShapeRule* rule);

} // namespace fx_converter
