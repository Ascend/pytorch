#pragma once

#include <cstdint>
#include <vector>

#include "graph/graph.h"
#include "exe_graph/fx_node.h"

namespace fx_converter {

class FxGraphBuilder;

ge::graphStatus RewriteTypedConstNodes(
    ge::Graph* graph,
    FxGraphBuilder* builder,
    std::vector<Node*>* kernel_nodes,
    uint64_t& sym_gen);

} // namespace fx_converter
