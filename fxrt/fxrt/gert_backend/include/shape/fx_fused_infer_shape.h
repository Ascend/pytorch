#pragma once

#include <cstddef>
#include <vector>

#include "shape/fx_shape_rule.h"

namespace fx_converter {

// RT node type of the fused InferShape kernel (see fx_fused_infer_shape.cc).
constexpr const char* kFxFusedInferShapeType = "FxFusedInferShape";

// One entry per fused output slot: which rule produces that slot's shape. Multi-output ops
// (TopKV2 values/indices) repeat the same rule pointer in consecutive slots, matching what the
// per-op kernel did by mirroring one rule onto every output.
//
// Owned by BuiltExecuteGraph (kept alive for the life of the ExecuteGraph); the Main graph holds
// only an 8-byte Const pointing here, exactly like FxShapeRule*.
struct FxFusedShapeRules {
  std::vector<const FxShapeRule*> slot_rules;
  // Trailing output slots copy the current feed's logical shape for fallback.
  // A feed descriptor can contain unknown dimensions without a symbol seed;
  // its concrete shape is already available in the runtime input tensor.
  std::vector<size_t> feed_shape_indices;
};

} // namespace fx_converter
