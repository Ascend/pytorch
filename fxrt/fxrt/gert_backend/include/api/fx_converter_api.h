#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/runtime/tensor.h" // CANN
#include "ir/graph.h"

namespace fx_converter {

class EsGraphRunner;

// Deleter handed to the consumer along with a GE-allocated output buffer.
// Plain C function pointer so it can cross the ABI=0 / ABI=1 boundary as an integer.
using FxFreeOwnedFn = void (*)(void*);

struct InputTensorLayout {
  std::vector<int64_t> logical_shape;
  std::vector<int64_t> storage_shape;
  std::vector<int64_t> stride;
  // Element offset from input_addrs' storage base. The runtime TensorV2 carries
  // this value for consumers that support strided tensors.
  int64_t storage_offset = 0;
  // ge::Format / aclFormat. 2 is FORMAT_ND.
  int32_t format = 2;
};

// One physical model output after ExecuteAlloc, with its logical return layouts.
//
// Ownership: GE allocates the buffer from its L1 allocator (AllocModelOutTensor with a
// null output address) and hands it to us via gert::TensorData. ExecuteAlloc then calls
// TensorData::Release, so the buffer belongs to *the caller*, not to this session.
// The caller must eventually call free_fn(owned_ctx) exactly once — attaching it as a
// torch Tensor deleter is the intended use. owned_ctx == nullptr means GE did not own
// the memory (caller-supplied address); in that case there is nothing to free.
struct ExecutedOutput {
  std::vector<int64_t> shape;
  // Physical dims. Empty means the same as `shape`.
  std::vector<int64_t> storage_shape;
  int32_t dtype = 0; // ge::DataType
  // ge::Format / aclFormat of the storage. 2 is FORMAT_ND.
  int32_t format = 2;
  uintptr_t device_addr = 0;
  size_t nbytes = 0; // storage bytes: prod(storage_shape) * sizeof(dtype)
  // Logical stride of an adopted fallback view. Empty means contiguous.
  std::vector<int64_t> stride;
  void* owned_ctx = nullptr;
  FxFreeOwnedFn free_fn = nullptr;
  std::vector<OutputViewSpec> logical_views;
};

// Deleter for ExecutedOutput::owned_ctx. Returns the block to the GE L1 allocator and
// drops the ES graph runner keep-alive reference. Exported so the ABI=1 torch bridge can
// call it through a function pointer without linking against GE.
extern "C" __attribute__((visibility("default"))) void FxFreeOwnedOutput(void* ctx);

// Convert only: ABI=0 fxrt::ir::Graph -> BuiltExecuteGraph.
// Debug (no control-flow change): FX_CONVERTER_DUMP_GRAPH prints the built graph as text.
BuiltExecuteGraph ConvertFromMrtIr(const fxrt::ir::GraphPtr& graph, const std::string& name);

// Convert + initialize once; Execute many times (torch.compile backend session).
class ConverterSession {
 public:
  ConverterSession() = default;
  ~ConverterSession();

  ConverterSession(const ConverterSession&) = delete;
  ConverterSession& operator=(const ConverterSession&) = delete;
  ConverterSession(ConverterSession&&) noexcept = default;
  ConverterSession& operator=(ConverterSession&&) noexcept = default;

  // Convert ABI=0 IR and initialize the ES graph runner. Returns false on failure (see *err).
  // `stream` is the caller's Execute stream (torch's current stream). ES execution
  // synchronizes it before GE reads input buffers.
  bool Convert(
      const fxrt::ir::GraphPtr& graph,
      const std::string& name,
      int32_t device_id,
      void* stream,
      std::string* err);

  // GE-allocated output path: output Tensors are handed to the ES runner with a null address.
  // Fills *outputs and transfers ownership of every buffer to the caller (see ExecutedOutput).
  // The ES runner synchronizes the caller's stream before reading inputs, then returns
  // completed outputs.
  // scalar_syms carries the runtime values of symbolic scalars (Dynamo-lifted SymInt
  // placeholders) as (symbol name, value) pairs; empty when the graph has none.
  bool ExecuteAlloc(
      const std::vector<void*>& input_addrs,
      const std::vector<InputTensorLayout>& input_layouts,
      void* stream,
      std::vector<ExecutedOutput>* outputs,
      std::string* err,
      const std::vector<std::pair<std::string, int64_t>>& scalar_syms = {});

  // D2D copy helper for Stage 1 Python wrap (aclrtMemcpy DEVICE_TO_DEVICE).
  static bool MemcpyDeviceToDevice(void* dst, const void* src, size_t nbytes, std::string* err);

  void release();
  bool ready() const {
    return es_runner_ != nullptr;
  }
  bool has_es_runner() const {
    return es_runner_ != nullptr;
  }
  bool has_fallback_calls() const;
  size_t output_count() const {
    return output_dtypes_.size();
  }
  // GE DataType (int32) for outputs — used to allocate correctly (e.g. TopKV2 indices INT32).
  const std::vector<int32_t>& output_dtypes() const {
    return output_dtypes_;
  }

 private:
  // The runner owns the GE session and graph. Each ExecutedOutput::owned_ctx
  // holds a strong ref so the session outlives GE allocated output buffers.
  std::shared_ptr<EsGraphRunner> es_runner_;
  std::vector<int32_t> input_dtypes_;
  std::vector<bool> input_requires_base_format_;
  std::vector<int32_t> output_dtypes_;
  std::vector<std::vector<OutputViewSpec>> output_views_;
  int32_t device_id_ = 0;
};

} // namespace fx_converter
