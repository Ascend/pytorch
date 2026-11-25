#pragma once

#include <string>
#include <utility>
#include <vector>
#include "lowering/op_lowering.h"

namespace fx_converter {

// Runtime shape calculation keeps the existing FXRT graph. Static reshape
// geometry uses the common C++ helpers; the planner selects each edge's ABI.
struct EdgeLayoutPlan {
  std::vector<std::pair<std::string, TensorRef>> originals;
  std::vector<std::pair<std::string, TensorRef>> replacements;
  std::vector<LoweringResult> adapters;
};

// Installs consumer-local TensorRef substitutions. Restore after Lower, including failure.
// Adapters are ordinary lowerings; the caller attaches their CNI and memory lifetime.
bool PlanInputLayouts(
    const fxrt::ir::NodePtr& node,
    const OpLowering& lowering,
    const std::string& tag,
    LoweringContext& ctx,
    EdgeLayoutPlan* plan,
    std::string* error);
void RestoreInputLayouts(LoweringContext& ctx, const EdgeLayoutPlan& plan);
void ApplyInputLayouts(LoweringContext& ctx, const EdgeLayoutPlan& plan);

// Builds a borrowed dense address at source + element_offset * item_size.
// Used by dense input ABIs and by the existing copy-out boundary. The caller
// must retain result.allocation until its last reader; no device storage is allocated.
bool RebaseContiguousView(
    const TensorRef& source,
    const std::string& tag,
    LoweringContext& ctx,
    TensorRef* result,
    std::string* error);

} // namespace fx_converter
