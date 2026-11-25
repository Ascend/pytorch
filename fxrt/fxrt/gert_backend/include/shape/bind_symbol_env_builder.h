#pragma once

#include <vector>

#include "shape/fx_symbol_env.h"
#include "exe_graph/graph_builder_utils.h"
#include "ir/graph.h"

namespace fx_converter {

// Attach graph-level BindSymbolEnv before InferShape (shared SymbolEnv).
fx_converter::Node* AttachBindSymbolEnv(
    ge::Graph* main,
    FxSymbolEnv* env,
    const std::vector<fx_converter::Node*>& graph_input_datas,
    const std::vector<fxrt::ir::NodePtr>& graph_inputs,
    uint64_t& sym_gen,
    std::vector<fx_converter::Node*>* kernel_nodes);

} // namespace fx_converter
