#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "shape/fx_symbol_env.h"
#include "ir/symbolic/symbolic.h"

// InferShape rule: output dims evaluated against a shared FxSymbolEnv.
// out_dims_ir is the fxrt::ir expression tree. InferShape and OM2 codegen both read it.
struct FxShapeRule {
  std::vector<fxrt::ir::SymbolicExprPtr> out_dims_ir;
  // The output is a genuine rank-0 tensor (argmax without keepdim, a full reduce):
  // empty out_dims_ir is the correct rule, not a missing one. Lets the evaluators accept
  // the empty vector they would otherwise reject as broken.
  bool is_rank0 = false;
  // Shared env (owned by BuiltExecuteGraph). Required for Evaluate.
  fx_converter::FxSymbolEnv* env = nullptr;
  std::string source = "unset";
};
