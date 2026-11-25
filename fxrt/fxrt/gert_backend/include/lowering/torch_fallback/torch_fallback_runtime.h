#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "exe_graph/runtime/eager_op_execution_context.h"
#include "ir/value/value.h"
#include "shape/fx_symbol_env.h"
#include "torch_fallback/torch_fallback_abi.h"

namespace fx_converter {

constexpr size_t kNoTorchFallbackTensor = static_cast<size_t>(-1);

// Host scratch reused across Execute (defined in the runtime TU).
struct TorchFallbackRuntimeCache;

enum class FallbackShapeExprKind : uint8_t {
  kConstant,
  kDimension,
  kAdd,
  kMul,
  kTrueDiv,
  kFloorDiv,
  kCeilDiv,
  kMod,
  kMin,
};

struct FallbackShapeExpr {
  FallbackShapeExprKind kind = FallbackShapeExprKind::kConstant;
  int64_t value = 0;
  size_t feed_index = 0U;
  size_t dim_index = 0U;
  std::shared_ptr<FallbackShapeExpr> lhs;
  std::shared_ptr<FallbackShapeExpr> rhs;
};

struct TorchFallbackArgument {
  fxrt::ir::ValuePtr value;
  size_t tensor_index = kNoTorchFallbackTensor;
  // NPU private storage format id (aclFormat, e.g. FRACTAL_NZ=29) carried in
  // FxTorchTensorView.reserved so the bridge tags the rebuilt tensor. 0 = ND.
  uint32_t tensor_format = 0U;
  // Physical FRACTAL_NZ layout captured from the IR tensor at Convert time.
  // GE runtime StorageShape is often a 1-D nbytes blob and cannot be used as-is.
  std::vector<int64_t> ir_storage_sizes;
  // Logical shape and stride of this IR tensor. A view alias's GE tensor keeps
  // the parent storage shape, so the fallback rebuilds sizes from here.
  std::vector<int64_t> ir_logical_shape;
  std::vector<int64_t> ir_stride;
  // Recursive children preserve Tensor[] / scalar-list structure while tensor
  // leaves are still wired to GE addresses by tensor_index.
  std::vector<TorchFallbackArgument> elements;
  // Indices into TorchFallbackRuntimeCache shape / list arenas. Convert-time.
  size_t runtime_shape_slot = kNoTorchFallbackTensor;
  size_t runtime_list_slot = kNoTorchFallbackTensor;
  // Runtime symbolic integers are evaluated from explicit shape-tensor inputs.
  std::shared_ptr<FallbackShapeExpr> shape_expr;
};

// One model feed visible to fallback for the current Execute.
struct FallbackFeed {
  uintptr_t addr = 0;
  int32_t format = 2;
  std::vector<int64_t> logical_shape;
  std::vector<int64_t> storage_shape;
  std::vector<int64_t> stride;
  int64_t storage_offset = 0;
  bool contiguous = true;
};

// How the runtime handles a deferred result whose storage overlaps an input
// without a schema ref pair (a dynamic ref). Set by the Create* entry of each
// fallback kind; never switch on the api table identity for this.
enum class FallbackDynamicRefPolicy : uint32_t {
  // custom_call: a dynamic ref means the op hid an alias; reject the output.
  kReject = 0,
  // python_call: copy a standard-layout (contiguous, offset 0, base format)
  // dynamic ref into a fresh GE output; reject anything else.
  kCopyIfStandard = 1,
};

// Per-model fallback execution state. Owned by BuiltExecuteGraph and shared
// with every TorchFallbackCall of the same graph, so the kernel side (record)
// and the API side (publish / consume) see one copy instead of process-wide
// globals. ConverterSession serializes Execute, and kernels run on the Execute
// thread, so no internal synchronization is needed.
class FallbackSessionState {
 public:
  FallbackSessionState() = default;
  ~FallbackSessionState(); // flushes adopted results
  FallbackSessionState(const FallbackSessionState&) = delete;
  FallbackSessionState& operator=(const FallbackSessionState&) = delete;

  // Live feeds for the current Execute, matched to rebuilt tensors by address.
  void PublishFeeds(std::vector<FallbackFeed> feeds) {
    feeds_ = std::move(feeds);
  }
  const std::vector<FallbackFeed>& Feeds() const {
    return feeds_;
  }

  // Deferred output layouts recorded by the kernels of the current Execute,
  // keyed by device address. Cleared at the start of each Execute so a stale
  // entry can never tag an output that sits on a recycled buffer.
  void ClearOutputLayouts() {
    output_layouts_.clear();
  }
  void RecordOutputLayout(
      void* addr,
      int32_t format,
      std::vector<int64_t> storage_shape,
      std::vector<int64_t> stride,
      std::vector<int64_t> logical_shape);
  bool LookupOutputLayout(
      const void* addr,
      int32_t* format,
      std::vector<int64_t>* storage_shape,
      std::vector<int64_t>* stride,
      std::vector<int64_t>* logical_shape) const;

  // Torch storages adopted by the current Execute. The graph outputs handed to
  // the caller are borrowed views of those storages, so they stay alive until
  // the next Execute starts (or the session is destroyed).
  std::shared_ptr<void> KeepAdoptedResult(const FxTorchFallbackApi* api, uintptr_t handle);
  void RecordOutputOwner(const void* addr, const std::shared_ptr<void>& owner);
  std::shared_ptr<void> LookupOutputOwner(const void* addr) const;
  void FlushAdoptedResults();

 private:
  std::vector<FallbackFeed> feeds_;
  std::vector<std::pair<uintptr_t, FallbackFeed>> output_layouts_;
  std::vector<std::shared_ptr<void>> adopted_results_;
  std::vector<std::pair<uintptr_t, std::weak_ptr<void>>> output_owners_;
};

struct TorchFallbackCall {
  // ---- Convert-time configuration (finalized during lowering) ----
  const FxTorchFallbackApi* api = nullptr;
  void* plan = nullptr;
  std::string qualified_name;
  // GE proto identity includes the call's port/alias contract, independently of dispatcher identity.
  std::string proto_type;
  std::string overload_name;
  std::vector<TorchFallbackArgument> arguments;
  size_t tensor_count = 0U;
  size_t shape_input_count = 0U;
  std::vector<std::vector<int64_t>> shape_input_shapes;
  std::vector<int32_t> shape_input_dtypes;
  std::vector<int32_t> output_dtypes;
  std::vector<std::vector<int64_t>> output_shapes;
  FxSymbolEnv* symbol_env = nullptr;
  // PythonCall publishes dense output buffers. These flags come from the FX FakeTensor layout.
  std::vector<int8_t> python_output_dense;
  // Schema-declared read aliases paired with the input argument and output index.
  std::vector<std::pair<size_t, size_t>> read_alias_pairs;
  // Outputs whose functional contract proves a fresh dense allocation.
  std::vector<int8_t> fresh_dense_outputs;
  // (schema argument index, output index) pairs for mutating (out-param)
  // ops: a write-alias Tensor argument (e.g. vllm.dsa_forward's "output"
  // argument) paired with the return that aliases it. Drives GE's
  // output-ref-input reuse marking (MakeOutputRefInput) at execution time.
  std::vector<std::pair<size_t, size_t>> ref_pairs;
  // Runtime-sized outputs use the deferred ABI: execute functionally, then
  // adopt the Torch storages into GE output shells (no second L1 malloc).
  // Ref-paired outputs are the exception: they alias their write-alias input
  // via MakeOutputRefInput instead of adopting a result storage.
  bool deferred_shape = false;
  // Undeclared dynamic-ref handling for deferred outputs. Schema read aliases
  // are handled separately; a proven fresh-dense whole-input result may be
  // copied when a functional bridge unexpectedly returns the input storage.
  // This policy describes the contract, not api-table identity.
  FallbackDynamicRefPolicy dynamic_ref_policy = FallbackDynamicRefPolicy::kReject;
  // Per-model Execute state shared with the sibling calls of the same graph.
  std::shared_ptr<FallbackSessionState> session;

  // ---- Per-call Execute scratch (mutated only from the serialized kernel Execute) ----
  // Cached aclrtGetDevice result; -1 means unset (device 0 is valid).
  int32_t cached_device_id = -1;
  std::unique_ptr<TorchFallbackRuntimeCache> cache;

  ~TorchFallbackCall();
};

// Fallback feed and alias boundary. custom_call and python_call share this.
//
// Format. The live aclFormat and storage shape are forwarded onto the Torch
// tensor the kernel sees. A non-contiguous feed whose format is not a base
// format (ND, NCHW, NHWC, NCDHW) is rejected. A private-format fallback
// result is rejected immediately. It is not relabeled ND for a later op
// or for the graph output.
//
// Stride. A base-format graph input is rebuilt with its live stride and the
// feed's storage-base pointer plus element storage_offset.
// An in-graph view keeps the IR stride when that stride disagrees with the
// feed. custom_call adopts a non-contiguous base-format result with that
// stride and does not copy it. python_call rejects a non-standard layout.
// Unproved or negative strides stay rejected.
//
// Alias. Mutation ref pairs and proven same-address dense read aliases reuse
// their paired input. An undeclared whole-input result can be copied only
// under a proven fresh-dense functional contract or PythonCall's standard
// layout policy. Partial aliases and unknown layout contracts remain rejected.

bool FallbackFeedIsBaseFormat(int32_t storage_format);

// True when shape/stride are a dense row-major layout. Empty stride is contiguous.
bool FallbackStrideIsContiguous(const int64_t* shape, const int64_t* stride, size_t rank);

const char* FallbackNoncontiguousPrivateReason();

bool RegisterTorchFallbackApi(uintptr_t api_address, std::string* error);

bool RegisterPythonCallApi(uintptr_t api_address, std::string* error);

std::shared_ptr<TorchFallbackCall> CreatePythonCall(
    const std::string& module_name,
    const std::string& func_name,
    std::vector<TorchFallbackArgument> arguments,
    FxSymbolEnv* symbol_env,
    std::vector<int32_t> output_dtypes,
    std::shared_ptr<FallbackSessionState> session,
    std::string* error);

// Functional plans cannot reuse a WithInferShape template: GE will not realloc
// once the template has already set TensorData size. Out= plans keep the
// InferShape fast path because they write into that allocation.
bool TorchFallbackPlanUsesOutSchema(const TorchFallbackCall& call);

std::shared_ptr<TorchFallbackCall> CreateTorchFallbackCall(
    const std::string& qualified_name,
    const std::string& overload_name,
    std::vector<TorchFallbackArgument> arguments,
    FxSymbolEnv* symbol_env,
    std::vector<int32_t> output_dtypes,
    std::vector<std::pair<size_t, size_t>> read_alias_pairs,
    std::vector<int8_t> fresh_dense_outputs,
    std::shared_ptr<FallbackSessionState> session,
    std::string* error);

// Runs one fallback call. The kernel context must use EagerOpExecutionContext's
// additional-input layout: allocator, then stream, then this call pointer.
ge::graphStatus RunTorchFallback(TorchFallbackCall* call, gert::EagerOpExecutionContext* ctx);

} // namespace fx_converter
