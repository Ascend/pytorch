#pragma once

#include <string>
#include <vector>

#include "ir/graph.h"
#include "ir_import/ir_graph_spec.h"

namespace fx_converter {

// Materialize ABI=0 fxrt::ir::Graph from a structured IR snapshot (Python export).
fxrt::ir::GraphPtr BuildMrtIrGraphFromSpec(const IrGraphSpec& spec);

} // namespace fx_converter
