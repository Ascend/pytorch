#pragma once

#include <limits>
#include <string>
#include <vector>

#include "lowering/aicore/aicore_bin_artifacts.h"
#include "lowering/aicore/aicore_rt_wiring.h"
#include "lowering/catalog_attrs.h"
#include "lowering/op_lowering.h"
#include "register/op_catalog.h"

namespace fx_converter {

// Effective OPP proto when it is not the catalog row. Generic ops copy type/names
// and keep pack_entry on that row. AsStrided rewrites type/names, caps the select
// key, and drops pack_entry so Slice attrs are not applied to the AsStrided tiling op description.
struct AiCoreOppBinding {
  const char* opp_type = nullptr;
  const char* opp_input_names = nullptr;
  // Cap on the select input slots; 0 keeps every launch input.
  size_t select_input_count = 0;
  const OpCatalogEntry* pack_entry = nullptr;
};

struct AiCoreOpWork {
  const fxrt::ir::NodePtr& node;
  const OpCatalogEntry& entry;
  const std::string& tag;
  std::vector<TensorRef> ins;
  int64_t axis = std::numeric_limits<int64_t>::min();
  CatalogAttrExtras extras;
  std::vector<std::vector<int64_t>> out_shapes;
  std::vector<ge::DataType> dtypes;
  AiCoreOppBinding bind;
  AiCoreBinArtifacts art;
  AiCoreOptionalLayout optional;
  std::vector<bool> final_mask;
  AiCoreStage stage;
};

} // namespace fx_converter
