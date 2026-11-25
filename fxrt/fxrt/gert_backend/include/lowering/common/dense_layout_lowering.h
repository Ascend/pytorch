#pragma once

#include "lowering/op_lowering.h"

namespace fx_converter {

// Called only after the planner has decided that a dense value is needed.
// A non-match leaves the graph unchanged and uses the general layout path.
// A matched implementation failing to lower is an error, not a partial fallback.
bool TryLowerSpecializedDenseLayout(
    const fxrt::ir::NodePtr& input,
    const TensorRef& source,
    const std::string& tag,
    LoweringContext& ctx,
    LoweringResult* result,
    bool* matched,
    std::string* error);

} // namespace fx_converter
