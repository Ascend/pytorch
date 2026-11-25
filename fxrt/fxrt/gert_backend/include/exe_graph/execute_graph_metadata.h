#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "graph/graph.h"
#include "lowering/catalog_attrs.h"

namespace fx_converter {
class Node;
} // namespace fx_converter

// Mix launch reusable resource counts written to gert.model.stream_count / notify_count
// (GE LaunchMixKernelWithHandle needs stream[1] + notify[2]).
constexpr uint32_t kReusableStreamNum = 2U;
constexpr uint32_t kReusableNotifyNum = 2U;

struct ModelIoSpec {
  std::string name;
  std::vector<int64_t> shape;
  int32_t data_type = 0; // ge::DataType; prefer IR-derived value at Convert time
  ge::Format format = ge::FORMAT_ND;
  // Physical size. Empty means the same as `shape`.
  std::vector<int64_t> storage_shape;
};

// Generic CNI description for one logical op (OPP type + IO + linked RT nodes).
// Per-op GE attributes live in attrs. Add a field there, not a new member here.
struct OpCniSpec {
  std::string op_type;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<int32_t> input_dtypes; // ge::DataType
  std::vector<int32_t> input_formats; // ge::Format; empty means FORMAT_ND
  std::vector<int32_t> output_formats; // ge::Format; empty means FORMAT_ND
  std::vector<int64_t> output_shape;
  int32_t output_dtype = 0; // ge::DataType
  // Multi-output: when non-empty, overrides single output_shape/dtype for CNI.
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int32_t> output_dtypes; // ge::DataType
  std::vector<fx_converter::Node*> compute_nodes;
  // Optional axis / concat_dim. INT64_MIN means unset.
  int64_t axis = std::numeric_limits<int64_t>::min();
  fx_converter::CniIoStyle cni_io = fx_converter::CniIoStyle::kIndexedFrom0;
  const fx_converter::OpCatalogEntry* catalog = nullptr;
  const fx_converter::OpBehavior* behavior = nullptr;
  // Optional comma-separated OPP IR input names (empty → x0, x1, ...).
  std::string opp_input_names;
  // Optional comma-separated OPP IR output names (e.g. values,indices).
  std::string opp_output_names;
  // Per-op IR-collected attr slots (not a graph-wide table).
  fx_converter::CatalogAttrExtras attrs;
};

bool AttachModelV2Metadata(
    ge::Graph* root,
    const std::vector<fx_converter::Node*>& kernel_nodes,
    const std::vector<OpCniSpec>& op_specs,
    const std::vector<ModelIoSpec>& inputs,
    const std::vector<ModelIoSpec>& outputs);
