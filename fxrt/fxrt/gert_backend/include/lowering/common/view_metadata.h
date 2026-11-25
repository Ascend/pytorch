#pragma once

#include "exe_graph/graph_builder_utils.h"
#include "ir/graph.h"
#include "shape/fx_symbol_env.h"

namespace fx_converter {
struct NativeViewLayout {
  std::vector<int64_t> size;
  std::vector<fxrt::ir::SymbolicExprPtr> size_exprs;
  std::vector<int64_t> stride;
  std::vector<fxrt::ir::SymbolicExprPtr> stride_exprs;
  int64_t relative_offset = 0;
  fxrt::ir::SymbolicExprPtr relative_offset_expr;
  bool aliases = false;
  bool contiguous = false;
};

// Read a PyTorch-inferred candidate only if its input matches the selected
// producer's actual layout. Dense producers select the dense-input candidate.
bool ReadNativeViewLayout(
    const fxrt::ir::NodePtr& node,
    const TensorRef& source,
    FxSymbolEnv* env,
    NativeViewLayout* layout);
} // namespace fx_converter
