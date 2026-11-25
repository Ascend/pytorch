#pragma once

#include <memory>
#include <string>
#include <vector>

#include "lowering/aicore/aicore_binary_kernel_selector.h"
#include "shape/fx_shape_rule.h"
#include "shape/fx_fused_infer_shape.h"
#include "shape/fx_symbol_env.h"
#include "ge/es_c_graph_builder.h"
#include "exe_graph/es_graph_builder.h"
#include "ir/graph.h"
#include "lowering/torch_fallback/torch_fallback_runtime.h"

namespace fx_converter {
// A logical return backed by one physical GE output. preserve_layout=false
// uses that output's runtime dense shape without reconstruction.
struct OutputViewSpec {
  size_t output_index = 0;
  bool preserve_layout = false;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  int64_t storage_offset = 0;
  // Evaluated after Execute using the same bound symbols as InferShape.
  std::vector<fxrt::ir::SymbolicExprPtr> shape_exprs;
  std::vector<fxrt::ir::SymbolicExprPtr> stride_exprs;
  fxrt::ir::SymbolicExprPtr offset_expr;
  bool derive_dense_stride = false;
};
} // namespace fx_converter

struct BuiltExecuteGraph {
  // Logical execution graph under construction. The ES builder owns the Graph object;
  // logical_graph is a non-owning view that remains valid until the builder is reset.
  // The graph is finalized with EsCGraphBuilder::BuildGraphAndReset in the next migration step.
  std::unique_ptr<::EsCGraphBuilder> es_builder;
  std::unique_ptr<fx_converter::FxGraphBuilder> fx_builder;
  ge::Graph* logical_graph = nullptr;

  // Transitional runtime graph fields. These are no longer initialized by
  // CreateGraphSkeleton and will be removed as lowerings move to the ES graph.
  fx_converter::EsGraphRef root;
  fx_converter::EsGraphRef main;
  fx_converter::EsGraphRef init;
  fx_converter::EsGraphRef deinit;

  // Shared free-var env for all InferShape nodes.
  std::shared_ptr<fx_converter::FxSymbolEnv> symbol_env;
  // Keep InferShape Expression rules alive for ModelV2 execution.
  std::vector<std::shared_ptr<FxShapeRule>> shape_rules;
  // Slot -> rule table for the fused InferShape node (null when fusion did not run).
  std::shared_ptr<fx_converter::FxFusedShapeRules> fused_shape_rules;
  // Keep AICore binary payloads (BinData blob + compile_info strings) alive for Const nodes.
  std::vector<fx_converter::SelectedBinaryKernel> binary_kernels;
  std::vector<std::shared_ptr<std::vector<uint8_t>>> binary_bindata_blobs;
  // Device buffers for Path-C scalar / shape-tensor materialization (aclrtFree via custom deleter).
  std::vector<std::shared_ptr<void>> aicore_device_bufs;
  // Keep ABI=1 dispatcher plans and ABI=0 argument metadata alive for fallback adapters.
  std::vector<std::shared_ptr<fx_converter::TorchFallbackCall>> torch_fallback_calls;
  // Per-model fallback execution state (feeds, adopted storages, deferred output
  // layouts); each TorchFallbackCall above holds a copy of this shared_ptr.
  std::shared_ptr<fx_converter::FallbackSessionState> fallback_session =
      std::make_shared<fx_converter::FallbackSessionState>();
  // Shape-list Const may contain PyTorch -1 (keep dim): the ABI0 int_list sentinel that
  // fx_to_c_graph.py writes for a SymInt dim. Refresh the host/device payload
  // each Execute before FxAiCoreTiling / Launch. The tiling StorageShape is NOT refreshed:
  // a Broadcast target reuses the live InferShape output, and a vector meta shape ([rank]) is
  // fixed at Convert time — both are plain graph values, not host-poked buffers.
  struct AiCoreShapeTensorRefresh {
    void* host = nullptr; // payload for device H2D (Launch shape Const), INT64 or INT32
    void* dev = nullptr; // kOnDeviceHbm bytes for LaunchKernel
    std::vector<int64_t> size_template; // symbolic axes hold kSymbolicDim; a literal -1 stays -1
    // Element type of the buffers above, mirroring MaterializeShapeTensor's payload_dtype
    // (DT_INT64 or DT_INT32). The baseline is the graph's own int64; a call site narrows to
    // INT32 only where a TBE binary forces it (TopKV2 k). The refresh must follow that choice --
    // writing int64 into a rank*4 buffer both corrupts the values and overruns the allocation.
    ge::DataType payload_type = ge::DT_INT64;
    // Per-axis symbolic expression, index-aligned with size_template: non-null exactly where the
    // graph gave a symbol instead of a number, and evaluated per Execute against the bound
    // FxSymbolEnv (the same seeds InferShape uses). Free vars are interned, so BindSymbolEnv's
    // SetValue reaches these trees. An entry exists only when at least one axis is symbolic.
    //
    // This is the whole resolution: a dim is either a number the graph wrote -- a literal -1
    // included -- or an expression evaluated here. Nothing is inferred from feed shapes, so a
    // graph whose feeds happen to multiply out to some other batch (ETA: a [4,16] embedding
    // table -> 64) can no longer silently size a kernel to it.
    std::vector<fxrt::ir::SymbolicExprPtr> axis_exprs;
    // Scalar operands and parameter vectors (Slice offsets, Transpose axes, etc.)
    // may contain zero or negative values. Only symbolic broadcast-target dimensions
    // require positivity here; each operator validates its own parameter semantics.
    bool allow_nonpositive = false;
  };
  std::vector<AiCoreShapeTensorRefresh> aicore_shape_refreshes;
  // Host int64 layout arrays referenced by FxBuildStridedTensor Const tensors.
  // Keep their storage alive for the lifetime of the cached execution graph.
  std::vector<std::shared_ptr<std::vector<int64_t>>> runtime_layout_payloads;
  // Dataset descriptors carry TensorV1. Rebuild stride-aware feed arguments
  // from the caller's live TensorV2 metadata without device parameter tensors.
  struct FeedLayoutRefresh {
    size_t feed_index;
    std::shared_ptr<std::vector<int64_t>> payload;
  };
  std::vector<FeedLayoutRefresh> feed_layout_refreshes;
  // Model feed/output dtypes (ge::DataType as int32) for Execute tensor construction.
  std::vector<int32_t> input_dtypes;
  std::vector<int32_t> output_dtypes;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  // Physical output slot -> logical returns. Owners are transferred once per
  // physical buffer, then Python creates metadata-only aliases on that owner.
  std::vector<std::vector<fx_converter::OutputViewSpec>> output_views;
  // ge::Format. Empty means every slot is FORMAT_ND.
  std::vector<int32_t> input_formats;
  std::vector<int32_t> output_formats;
  // Raw AICore readers require a base-format allocation, even when tracing
  // omitted a private storage format. ACLNN/fallback readers retain their ABI.
  std::vector<bool> input_requires_base_format;
  // Physical feed size. Empty or a short vector means the same as input_shapes.
  std::vector<std::vector<int64_t>> input_storage_shapes;
  // Host-only input layout state, refreshed before the synchronous ES Execute.
  // Const nodes store pointers to its fixed-size slots. Keep the allocation
  // stable and alive with the graph; offsets are metadata, not device inputs.
  std::shared_ptr<std::vector<int64_t>> input_storage_offsets;
  // Empty on success; non-empty means Convert failed (unsupported op, etc.).
  std::string error;
};

// Primary Converter: fxrt::ir::Graph (ABI=0) -> ExecuteGraph.
BuiltExecuteGraph BuildExecuteGraphFromMrtIr(const fxrt::ir::Graph& graph, const std::string& name);
