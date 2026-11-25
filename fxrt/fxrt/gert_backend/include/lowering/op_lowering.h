#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "exe_graph/execute_graph_metadata.h"
#include "exe_graph/graph_builder_utils.h"
#include "exe_graph/graph_skeleton_builder.h"
#include "ir/graph.h"
#include "shape/fx_symbol_env.h"

// Defined in execute_graph_builder.h (global); forward decl avoids heavy include here.
struct BuiltExecuteGraph;

namespace fx_converter {

// Cache SplitDataTensor by packed Tensor* so fan-out consumers share one unpack.
struct SplitCacheKey {
  const fx_converter::Node* producer = nullptr;
  uint64_t symbol = 0;
  bool operator==(const SplitCacheKey& o) const {
    return producer == o.producer && symbol == o.symbol;
  }
};
struct SplitCacheKeyHash {
  size_t operator()(const SplitCacheKey& k) const noexcept {
    return std::hash<const void*>()(static_cast<const void*>(k.producer)) ^ (std::hash<uint64_t>()(k.symbol) << 1U);
  }
};
struct SplitCacheEntry {
  fx_converter::Node* node = nullptr;
  uint64_t shape_symbol = 0;
  uint64_t addr_symbol = 0;
};

// A capability belongs to the selected lowering, not to a front-end op name.
// kStridedTensor additionally requires an ABI that actually transports strides.
// kMatrix accepts an unchanged runtime Tensor(V2) feed, or an internal view whose
// last-two transpose can be represented by the matrix ABI's transpose flags.
enum class InputLayoutKind : uint8_t { kDense, kMetadata, kMatrix, kStridedTensor, kStorageIndexed };

struct DenseLayoutValue {
  TensorRef tensor;
  fx_converter::Node* launch = nullptr;
};

// Shared Convert state passed into every OpLowering::Lower call.
struct LoweringContext {
  ge::Graph* main = nullptr;
  SharedRuntime* rt = nullptr;
  ::BuiltExecuteGraph* built = nullptr;
  FxSymbolEnv* symbol_env = nullptr;
  fx_converter::Node* bind_symbols = nullptr;
  const std::vector<fx_converter::Node*>* graph_input_datas = nullptr;
  uint64_t* sym_gen = nullptr;
  std::vector<fx_converter::Node*>* kernel_nodes = nullptr;
  std::unordered_map<std::string, TensorRef>* tensors = nullptr;
  std::unordered_map<const fxrt::ir::Node*, std::string>* name_of = nullptr;
  const std::vector<fxrt::ir::NodePtr>* compute_nodes = nullptr;
  // Graph returns are observable users when proving an allocation is exclusive.
  const std::vector<fxrt::ir::NodePtr>* return_values = nullptr;
  std::unordered_map<const fxrt::ir::Node*, std::vector<TensorRef>>* multi_out_tensors = nullptr;
  // Per-convert cache of OPP dual-stage Fwk nodes, keyed by OPP op type (e.g. "MatMul").
  std::unordered_map<std::string, DualStageOpFwk>* fwk_by_op = nullptr;
  // Init/DeInit resource symbols keyed by backend-provided resource identity.
  std::unordered_map<std::string, uint64_t>* autofuse_resources = nullptr;
  std::unordered_map<std::string, fx_converter::Node*>* autofuse_resource_nodes = nullptr;
  // Per-output-slot final flags, parallel to the node's output tensors. The builder fills
  // one entry per output; a slot is final when the graph returns that output (directly or
  // through one of its tuple_getitems). Single source of truth for final-ness.
  std::vector<bool> output_final_mask;
  // Nodes whose only consumers are compiled-kernel HOPs. Such a node's tensor is never read
  // by a GE op, and the kernel indexes it with strides baked in at codegen time, so a view
  // here has to stay a view onto the storage those strides address -- materializing it would
  // hand the kernel a dense buffer it then indexes with the base's strides.
  const std::unordered_set<const fxrt::ir::Node*>* compiled_kernel_only = nullptr;
  std::unordered_map<SplitCacheKey, SplitCacheEntry, SplitCacheKeyHash> split_cache;
  // Per-Convert values only. Never cache device data across Execute calls.
  std::unordered_map<std::string, DenseLayoutValue> dense_layout_cache;
};

// Final-ness of output slot 0 -- the only slot single-output lowerings have. Derived from
// output_final_mask (the single source of truth); false when the mask is empty.
inline bool IsFinalOutput(const LoweringContext& ctx) {
  return !ctx.output_final_mask.empty() && ctx.output_final_mask.front();
}

// Output of one successful Lower: tensor producers + control-edge anchors + CNI.
struct LoweringResult {
  // One TensorRef per output slot, in slot order; never empty on success. The builder
  // records the full vector so tuple_getitem consumers alias it natively.
  std::vector<TensorRef> outputs;
  fx_converter::Node* launch = nullptr;
  // Explicit prepare anchor for Launch->Prepare control chaining (do not use cni.compute_nodes[1]).
  fx_converter::Node* prepare = nullptr;
  // Intermediate AllocMemory nodes (GertTensorData*) of non-final slots, in slot order.
  // Freed after all consumer Launches (GE Guarder / last-consumer semantics), not after
  // the next op or final Launch.
  std::vector<fx_converter::Node*> mem_to_frees;
  OpCniSpec cni;
  // When true, skip appending cni (identity-style alias that registers no device op).
  bool skip_cni = false;
};

// Pluggable Convert strategy. Register via FX_REGISTER_OP_LOWERING in the .cc that defines it.
// Default path: DualStageOpLowering. Custom RT kernel sequences = new OpLowering.
class OpLowering {
 public:
  virtual ~OpLowering() = default;
  virtual const char* Name() const = 0;
  // IR op names (ops::ToStr form) this lowering handles that are not catalog rows.
  // Catalog-backed lowerings leave this empty; their names come from CatalogRegistry.
  virtual std::vector<std::string> SupportedLogicalOps() const {
    return {};
  }
  // First Match wins in registry priority order (lower priority value = earlier). ctx is nullptr
  // during the pre-dispatch support check (ValidateOpsSupported); the dispatch pass passes the
  // live context. Implementations that only care what the node is simply ignore ctx.
  virtual bool Match(const fxrt::ir::NodePtr& node, const LoweringContext* ctx = nullptr) const = 0;
  virtual InputLayoutKind InputLayout(const fxrt::ir::NodePtr&, size_t) const {
    return InputLayoutKind::kDense;
  }
  // Unknown Python/compiled effects invalidate cached layout snapshots.
  virtual bool MayWriteInputs() const {
    return false;
  }
  virtual bool Lower(
      const fxrt::ir::NodePtr& node,
      const std::string& tag,
      LoweringContext& ctx,
      LoweringResult* out,
      std::string* err) = 0;
};

// Common LoweringContext null checks. DualStage also requires fwk_by_op when require_fwk is true.
inline bool ValidateLoweringContext(const LoweringContext& ctx, bool require_fwk, const char* who, std::string* err) {
  if (ctx.main == nullptr || ctx.rt == nullptr || ctx.built == nullptr || ctx.tensors == nullptr ||
      ctx.name_of == nullptr || ctx.sym_gen == nullptr || ctx.kernel_nodes == nullptr ||
      (require_fwk && ctx.fwk_by_op == nullptr)) {
    if (err != nullptr) {
      *err = std::string(who != nullptr ? who : "lowering") + ": invalid context";
    }
    return false;
  }
  return true;
}

} // namespace fx_converter
