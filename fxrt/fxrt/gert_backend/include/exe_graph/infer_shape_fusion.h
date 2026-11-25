#pragma once

#include <cstdint>
#include <vector>

#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/execute_graph_metadata.h"

namespace fx_converter {

// Collapse every per-op InferShape node in Main onto one FxFusedInferShape node with one output
// slot per original output. Consumers keep their symbols and input indices; only the producing
// edge moves. Must run AFTER all lowering (so every InferShape exists) and BEFORE
// AttachModelV2Metadata (so the fused node gets gert.source_op and the removed nodes do not).
//
// Returns the number of InferShape nodes removed. 0 means the graph was left untouched -- the
// pass is all-or-nothing per graph and bails out on anything it does not fully recognise.
//
size_t FuseInferShapeNodes(
    BuiltExecuteGraph* built,
    fx_converter::Node* bind_symbols,
    std::vector<fx_converter::Node*>* kernel_nodes,
    std::vector<OpCniSpec>* cni_specs,
    uint64_t& sym_gen);

} // namespace fx_converter
