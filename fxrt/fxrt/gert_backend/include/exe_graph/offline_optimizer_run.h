#pragma once

#include "graph/ge_error_codes.h" // CANN
#include "graph/graph.h"

namespace fx_converter {

// Isolated from execute_graph_builder so OfflineOptimizer's inline ctor (which
// constructs hidden-visibility GE passes) is compiled with matching visibility
// and binds to gert_pass objects instead of CANN libgert dynsym.
ge::graphStatus RunOfflineOptimizer(ge::Graph* root);

} // namespace fx_converter
