#pragma once

#include <cstdint>
#include <vector>

#include "ir/graph.h"
#include "ir/value/value.h"
#include "shape/fx_symbol_env.h"

namespace fx_converter {

// Sentinel an ABI0 int_list carries for a dim whose value is only known at Execute.
// A list element is either a concrete Int -- including a literal -1 the FX graph itself wrote,
// e.g. reshape(x, [-1]) -- or a Symbol, and only the latter turns into this sentinel.
constexpr int64_t kSymbolicDim = -1;

// Read an IR int list without evaluating its symbols. Value::ToInt() calls Evaluate() on a
// Symbol, which throws while the env is unbound (i.e. all through Convert), so a bare
// Tuple::ToIntList() is only safe on a list proven concrete.
// Symbolic elements come back as kSymbolicDim.
std::vector<int64_t> ReadIntListSymbolSafe(const fxrt::ir::TuplePtr& tuple);

// Same, plus the per-axis expression: exprs[i] is the interned native tree for a symbolic
// element (ParseFxSymExpr, so BindSymbolEnv's SetValue reaches it) and nullptr for a concrete
// one. This is what lets Execute compute a dim instead of inferring it from feed shapes.
// Returns false when a symbolic element cannot be rebuilt into a native tree.
bool ReadIntListWithSymbols(
    const fxrt::ir::TuplePtr& tuple,
    FxSymbolEnv& env,
    std::vector<int64_t>* dims,
    std::vector<fxrt::ir::SymbolicExprPtr>* exprs);

// True when any element of the list is a Symbol (no evaluation).
bool IntListHasSymbol(const fxrt::ir::TuplePtr& tuple);

// Per-axis expressions of a tensor node's own symbolic shape, interned like the above.
// exprs[i] is null where that dim is a constant. This is what a size list seeded from a data
// tensor's shape (aten.slice) resolves its dynamic axes with, and what PyTorch's literal -1
// ("keep this dim") means for expand.
// Returns false when the node carries no symbolic shape.
bool SymbolicShapeExprsOfNode(
    const fxrt::ir::NodePtr& node,
    FxSymbolEnv& env,
    std::vector<fxrt::ir::SymbolicExprPtr>* exprs);

// Resolve PyTorch-style keep-dim markers (-1 / INT64_MIN) in a shape-list against
// an aligned data-tensor shape (right-aligned, like AclnnExpand / BroadcastTo).
// Returns false if rank/alignment is invalid or a keep-dim cannot be resolved.
bool ResolveKeepDimShapeList(std::vector<int64_t>* size, const std::vector<int64_t>& in_shape);

bool ResolveKeepDimShapeListStrict(std::vector<int64_t>* size, const std::vector<int64_t>& in_shape);

// Convert a 32-bit float scalar to IEEE 754 float16 bits (round-to-nearest-even).
// Used for Path C scalar constants stored as 16-bit device values, and by the
// Execute-time payload refresh for symbolic scalars materialized as fp16.
uint16_t Float32ToFp16Bits(float value);

// Convert a 32-bit float scalar to bfloat16 bits (truncate the lower mantissa).
uint16_t Float32ToBf16Bits(float value);

} // namespace fx_converter
