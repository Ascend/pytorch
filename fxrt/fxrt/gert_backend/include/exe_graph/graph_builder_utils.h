#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "graph/tensor.h"
#include "exe_graph/runtime/tensor.h" // GE_HEADER
#include "exe_graph/es_graph_builder.h"
#include "exe_graph/const_value_serialization.h"
#include "ir/symbolic/symbolic.h"

namespace fx_converter {

// The ES graph path uses the fused InferShape ABI and refreshes dynamic inputs per execute.
bool IsEsGraphRunnerEnabled();

constexpr int32_t kFeedStream = -1;
constexpr int32_t kFeedAllocator = -2;
constexpr int32_t kFeedEvents = -3;
constexpr int32_t kFeedNotifies = -4;

// One endpoint of a tensor flow: (producer node, output index, output symbol).
struct TensorEndpoint {
  fx_converter::Node* node = nullptr;
  int32_t out = 0;
  uint64_t sym = 0;

  bool Valid() const {
    return node != nullptr;
  }
};

// Unpacked Launch pair (GE: shape = InferShape/Split out0, addr = Alloc/Split out1).
struct UnpackedIo {
  TensorEndpoint shape;
  TensorEndpoint addr;

  bool Complete() const {
    return shape.Valid() && addr.Valid();
  }
};

struct TensorRef {
  fx_converter::Node* producer = nullptr;
  uint64_t symbol = 0;
  // Optional: host Tensor for FxAiCoreTiling (shape-list Broadcast target).
  // When invalid, tiling uses producer (Tensor starts with StorageShape).
  TensorEndpoint tiling_ep;
  // Host-owned AICore parameter. Launch must bind it through CopyFlowLaunch,
  // which embeds small values in launch args and copies larger values to device.
  bool host_parameter = false;
  // GE StorageShape. Always physical; never the ATen logical expand size.
  std::vector<int64_t> shape;
  // ATen layout. Empty stride means "treat as dense row-major of `shape`".
  std::vector<int64_t> stride;
  // Symbolic strides are evaluated from the current Execute's shape symbols.
  // `stride` stores placeholders for those slots; consumers must check this
  // vector before treating the concrete values as layout facts.
  std::vector<fxrt::ir::SymbolicExprPtr> stride_exprs;
  // Proof supplied by a dense-producing lowering; not an assumption about an
  // arbitrary symbolic layout. A dense dynamic layout may contain unknown strides.
  bool proven_dense = false;
  std::vector<int64_t> storage_shape;
  // ATen logical size for broadcast views. Empty means same as `shape`.
  std::vector<int64_t> logical_shape;
  // Symbolic logical dimensions, aligned with LogicalShape(ref).
  std::vector<fxrt::ir::SymbolicExprPtr> logical_shape_exprs;
  // Absolute element coordinate relative to unpacked.addr (the effective
  // storage anchor). Metadata views compose this coordinate without rebasing;
  // consumer address adaptation consumes it together with the runtime term.
  int64_t storage_offset = 0;
  // Symbolic offset replacing storage_offset at Execute time. The feed's runtime offset
  // remains a separate endpoint and is added only for relative view geometry.
  fxrt::ir::SymbolicExprPtr storage_offset_expr;
  // Per-Execute storage offset read from the external feed's host layout state. It is kept
  // separate from compile-time view geometry so cached graphs use the current feed.
  TensorEndpoint runtime_storage_offset;
  ge::DataType dtype = ge::DT_FLOAT;
  // Storage format. Origin format is the base layout of this value (ND for FRACTAL_NZ).
  ge::Format format = ge::FORMAT_ND;
  // Root feed slot for storage-sharing views; fresh allocations leave -1.
  // Used to validate the live storage format at selected raw-pointer ABIs.
  int32_t feed_index = -1;
  // Unpacked Launch pair. When complete, Path C Launch can skip a new SplitDataTensor.
  UnpackedIo unpacked;
  // Owner of a borrowed, offset-adjusted address. A pointer view allocates only
  // a host header; the original allocation must outlive all of its consumers.
  // Invalid means unpacked.addr itself is the allocation/lifetime endpoint.
  TensorEndpoint allocation;
  // Optional logical StorageShape for Autofuse / compiled_kernel (ATen size).
  TensorEndpoint logical_shape_ep;
  // Fallback results already own device memory that ExecuteAlloc can adopt.
  // Skip the extra AllocModelOut copy for those slots.
  bool can_adopt_for_model_output = false;

  bool HasUnpackedPair() const {
    return unpacked.Complete();
  }
  // Broadcast-target shape list (Expand): the broadcast target IS the op's output shape, so the
  // tiling view is wired from the live InferShape output instead of a refreshable Const. That
  // node does not exist yet at Collect time, so the host payload is parked here and the tiling
  // view is bound right after WireInferShape. Non-null marks "tiling view still pending".
  void* pending_tiling_host_ptr = nullptr;
  size_t pending_tiling_nbytes = 0;
};

// Base layout a private storage format expands from. FRACTAL_NZ expands from ND.
ge::Format OriginFormatOf(ge::Format storage_format);

// Storage format on the desc, origin format derived from it. Origin shape starts
// equal to `shape`; callers that know a distinct logical size overwrite it.
ge::TensorDesc MakeTensorDesc(const std::vector<int64_t>& shape, ge::DataType dtype, ge::Format storage_format);

ge::TensorDesc MakeNdTensorDesc(const std::vector<int64_t>& shape, ge::DataType dtype);

fx_converter::Node* AddFeedNode(
    ge::Graph* graph,
    const std::string& name,
    const char* node_type,
    int32_t index,
    const ge::TensorDesc& td,
    uint64_t& sym_gen);

fx_converter::Node* AddConstNode(
    ge::Graph* graph,
    const std::string& name,
    const void* data,
    size_t size,
    bool is_string,
    uint64_t& sym_gen,
    FxrtConstType const_type = FxrtConstType::kBytePtr,
    const char* stage = nullptr);

template <typename T>
fx_converter::Node* AddConstNode(
    ge::Graph* graph,
    const std::string& name,
    const T* data,
    size_t size,
    bool is_string,
    uint64_t& sym_gen,
    const char* stage = nullptr) {
  return AddConstNode(
      graph, name, static_cast<const void*>(data), size, is_string, sym_gen, InferFxrtConstType<T>(is_string), stage);
}

void WireSymbols(fx_converter::Node* node, const std::vector<uint64_t>& in_syms, const std::vector<uint64_t>& out_syms);

// Value-flow wrapper over ExecuteGraph wiring, mirroring GE's bg::ValueHolder pattern:
// a value is (producer node, output index, output symbol). Node() builds the OpDesc,
// allocates output symbols, wires input symbols, and adds data edges in input order, so
// callers keep one ordered value list instead of parallel symbol vectors plus per-edge
// index bookkeeping.
class RtGraphBuilder {
 public:
  // A builder value IS a tensor endpoint: same (node, out, sym) triple, so stage
  // endpoints and TensorRef endpoints feed RtGraphBuilder without field unpacking.
  using Value = TensorEndpoint;
  struct NodeResult {
    fx_converter::Node* node = nullptr;
    std::vector<Value> outs;
  };

  RtGraphBuilder(
      ge::Graph* main,
      const ge::TensorDesc& empty_td,
      uint64_t& sym_gen,
      std::vector<fx_converter::Node*>* kernel_nodes)
      : main_(main), empty_td_(empty_td), sym_gen_(sym_gen), kernel_nodes_(kernel_nodes) {}

  static Value From(fx_converter::Node* node, int32_t out, uint64_t sym) {
    return Value{node, out, sym};
  }
  static Value From(const TensorEndpoint& ep) {
    return ep;
  }

  // Const node; the payload is deep-copied by the Const OpDesc (GE: ValueHolder::CreateConst).
  Value Const(
      const std::string& name,
      const void* data,
      size_t size,
      bool is_string,
      FxrtConstType const_type = FxrtConstType::kBytePtr,
      const char* stage = nullptr);

  template <typename T>
  Value Const(const std::string& name, const T* data, size_t size, bool is_string, const char* stage = nullptr) {
    return Const(name, static_cast<const void*>(data), size, is_string, InferFxrtConstType<T>(is_string), stage);
  }

  // Node with ins.size() inputs (wired in order) and out_count outputs
  // (GE: ValueHolder::CreateDataOutput). Inputs and outputs use empty_td.
  NodeResult Node(const char* type, const std::string& name, const std::vector<Value>& ins, size_t out_count);

  // Same as Node(), but callers supply real tensor descriptors: input descs pair with
  // ins 1:1, output descs use td_outs[o] and fall back to empty_td when the vector is
  // shorter. Needed for nodes whose OpDesc carries dtype/shape (InferShape / OutputData).
  NodeResult TypedNode(
      const char* type,
      const std::string& name,
      const std::vector<Value>& ins,
      const std::vector<ge::TensorDesc>& td_ins,
      const std::vector<ge::TensorDesc>& td_outs);

  // Node whose producers use non-default output anchors (e.g. InferShape output o feeds
  // CalcTensorSize input 1). Symbol order still comes from ins.
  NodeResult SourcedNode(const char* type, const std::string& name, const std::vector<Value>& ins, size_t out_count);

  ge::Graph* main() const {
    return main_;
  }
  const ge::TensorDesc& empty_td() const {
    return empty_td_;
  }
  uint64_t& sym_gen() const {
    return sym_gen_;
  }
  std::vector<fx_converter::Node*>* kernel_nodes() const {
    return kernel_nodes_;
  }

 private:
  ge::Graph* main_;
  const ge::TensorDesc& empty_td_;
  uint64_t& sym_gen_;
  std::vector<fx_converter::Node*>* kernel_nodes_;
};

std::vector<int64_t> ShapeDims(const gert::Tensor* tensor);
std::vector<int64_t> ShapeDims(const gert::Tensor& tensor);

} // namespace fx_converter
