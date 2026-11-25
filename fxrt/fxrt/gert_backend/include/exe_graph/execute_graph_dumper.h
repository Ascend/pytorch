#pragma once

#include <string>

#include "exe_graph/execute_graph_builder.h"

namespace fx_converter {

// Dump BuiltExecuteGraph: subgraphs, type hist, InferShape wiring, SymbolEnv, shape rules.
// Used when FX_CONVERTER_DUMP_GRAPH is set (see ConvertFromMrtIr).
std::string DumpExecuteGraph(const BuiltExecuteGraph& built);

} // namespace fx_converter
