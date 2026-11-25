#pragma once

#include "exe_graph/graph_skeleton_builder.h"

namespace fx_converter {

// Resolve Expression Infer + OPP prepare/launch for an arbitrary OPP op type.
// Const nodes must reference process-long addresses (not stack).
bool ResolveOppDualStageBinding(const char* opp_type, OppDualStageFuncs* out);

} // namespace fx_converter
