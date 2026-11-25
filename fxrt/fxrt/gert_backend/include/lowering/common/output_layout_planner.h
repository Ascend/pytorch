#pragma once

#include "exe_graph/graph_builder_utils.h"
#include "ir/graph.h"

namespace fx_converter {

struct OutputViewSpec;

struct OutputStorageCandidate {
  TensorRef source;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  int64_t elements = 0;
  bool noncontiguous = false;
  // A clone may have been elided inside the graph, but independent returned
  // clones must still receive different output owners.
  const fxrt::ir::Node* isolation = nullptr;
};

// OM2 only. Prove a static, hole-free permutation/broadcast over graph-owned
// ND storage at offset zero. Unsupported geometry keeps the dense output ABI.
bool PlanOutputStorage(const fxrt::ir::NodePtr& node, const TensorRef& ref, OutputStorageCandidate* candidate);
bool SameOutputStorage(const OutputStorageCandidate& a, const OutputStorageCandidate& b);

// A returned dense root already exports its complete allocation. Reconstruct
// its returned views on that single owner, including dynamic sizes and holes.
bool PlanReturnedRootView(
    const fxrt::ir::NodePtr& root_node,
    const TensorRef& root,
    const fxrt::ir::NodePtr& node,
    const TensorRef& ref,
    OutputViewSpec* view);

} // namespace fx_converter
