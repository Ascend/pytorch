#pragma once

#include "shape/fx_symbol_env.h"
#include "ir/symbolic/symbolic.h"

namespace fx_converter {

// Rebuild a native fxrt::ir expression tree from an ABI-bridge symbolic dim.
//
// Why this exists: the export/import bridge is structure-lossy. ir_graph_export.cc emits each
// dim as SymbolicExpr::ToString(), and mrt_ir_graph_builder.cc wraps the whole string back into
// a single SymbolicVar -- so "(50 * s0)" arrives on the ABI=0 side as a *variable named*
// "(50 * s0)", not a tree. This recovers the tree so InferShape and OM2 codegen can use
// SymbolicExpr::Evaluate() and the printed form.
//
// Grammar is exactly what SymbolicExpr::ToString() emits -- fully parenthesized, so there is no
// precedence to resolve:
//   expr  := int | ident | '(' expr op expr ')' | fname '(' expr ',' expr ')'
//   op    := '+' | '*' | '/' | '%'
//   fname := 'floor_div' | 'ceil_div' | 'min'
//
// Free variables are interned through env->EnsureIrVar so that one BindSymbolEnv SetValue
// reaches every tree that mentions them (SymbolicVar carries its own value).
//
// Returns nullptr when the string does not match that grammar. The caller fails Convert.
fxrt::ir::SymbolicExprPtr ParseFxSymExpr(const fxrt::ir::SymbolicExprPtr& bridged, FxSymbolEnv& env);

} // namespace fx_converter
