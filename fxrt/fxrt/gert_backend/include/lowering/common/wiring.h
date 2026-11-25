#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "exe_graph/graph_builder_utils.h"
#include "lowering/op_lowering.h"

struct BuiltExecuteGraph;

namespace fx_converter {

struct OpCatalogEntry;

// Add an InferShape node and wire inputs, shape rule, and infer function.
// Returns the created node; writes the output shape symbol to *s_shape.
fx_converter::Node* AddInferShapeNode(
    ge::Graph* main,
    const std::string& tag,
    const std::vector<TensorRef>& inputs,
    const ge::TensorDesc& td_o,
    const ge::TensorDesc& empty_td,
    fx_converter::Node* c_rule,
    fx_converter::Node* c_infer,
    uint64_t& sym_gen,
    uint64_t* s_shape);

// Multi-output InferShape (GE: CreateDataOutput count = node out anchors). Writes one symbol per out.
fx_converter::Node* AddInferShapeNode(
    ge::Graph* main,
    const std::string& tag,
    const std::vector<TensorRef>& inputs,
    const std::vector<ge::TensorDesc>& td_outs,
    const ge::TensorDesc& empty_td,
    fx_converter::Node* c_rule,
    fx_converter::Node* c_infer,
    uint64_t& sym_gen,
    std::vector<uint64_t>* s_shapes);

// Add an OutputData node for a graph-final op output.
fx_converter::Node* AddOutputDataNode(
    ge::Graph* main,
    const std::string& tag,
    const ge::TensorDesc& td_o,
    uint64_t& sym_gen,
    uint64_t* s_out_tensor);

// dtype then shape. The shape port selects which InferShape output to size (TopKV2 values/indices).
struct CalcTensorSizeInputs {
  TensorEndpoint dtype;
  TensorEndpoint shape;
};

// Add a CalcTensorSizeFromStorage node. The dtype symbol may come from SharedRuntime or a local Const.
fx_converter::Node* AddCalcTensorSizeNode(
    ge::Graph* main,
    const std::string& tag,
    const CalcTensorSizeInputs& inputs,
    uint64_t& sym_gen,
    uint64_t* s_size);

// Capacity used for an ES model output whose runtime shape can exceed its
// compile-time descriptor. Keep this in sync with EsGraphRunner's output buffers.
size_t EsModelOutputCapacityBytes(const std::vector<int64_t>& shape, ge::DataType dtype);

// Add an AllocMemory node for an intermediate tensor.
fx_converter::Node* AddAllocMemoryNode(
    ge::Graph* main,
    const std::string& tag,
    const ge::TensorDesc& empty_td,
    uint64_t s_allocator,
    fx_converter::Node* select_l2,
    uint64_t s_size,
    fx_converter::Node* calc_size,
    uint64_t& sym_gen,
    uint64_t* s_addr);

// shape and address. Dtype, placement, and formats are scalar Consts created with the node.
struct BuildTensorInputs {
  TensorEndpoint shape;
  TensorEndpoint addr;
};

// dtype, placement, storage format, origin format. Each value is an int32 Const.
struct TensorFieldEndpoints {
  TensorEndpoint dtype;
  TensorEndpoint placement;
  TensorEndpoint storage_format;
  TensorEndpoint origin_format;

  bool Valid() const {
    return dtype.Valid() && placement.Valid() && storage_format.Valid() && origin_format.Valid();
  }
};

TensorFieldEndpoints AddTensorFieldConsts(
    ge::Graph* main,
    const std::string& tag,
    ge::DataType dtype,
    gert::TensorPlacement placement,
    ge::Format storage_format,
    uint64_t& sym_gen,
    std::vector<fx_converter::Node*>* kernel_nodes = nullptr);

// Add an FxBuildTensor node that produces a Tensor* for the next consumer.
fx_converter::Node* AddBuildTensorNode(
    ge::Graph* main,
    const std::string& tag,
    const ge::TensorDesc& td_o,
    const BuildTensorInputs& inputs,
    ge::DataType dtype,
    ge::Format storage_format,
    uint64_t& sym_gen,
    uint64_t* s_out_tensor,
    std::vector<fx_converter::Node*>* kernel_nodes = nullptr);

// AllocModelOutTensor for graph-final outputs (user buffer or GE/L1 when addr is null).
fx_converter::Node* AddAllocModelOutTensorNode(
    ge::Graph* main,
    const std::string& tag,
    const ge::TensorDesc& td_o,
    const ge::TensorDesc& empty_td,
    uint64_t s_allocator,
    fx_converter::Node* select_l2,
    uint64_t s_size,
    fx_converter::Node* calc_size,
    uint64_t s_out_tensor,
    fx_converter::Node* out_tensor_node,
    uint64_t& sym_gen,
    uint64_t* s_out_addr);

// FxEnsureTensorAtOutMemory inputs. Shape and address ports are not always 0.
struct EnsureOutMemoryInputs {
  TensorEndpoint shape;
  TensorEndpoint tensor_data;
  TensorFieldEndpoints fields;
  TensorEndpoint stream;
  TensorEndpoint output_data;
  TensorEndpoint allocator;
};

// After Launch: write InferShape onto OutputData Tensor (required for ExecuteAlloc / from_blob).
fx_converter::Node* AddEnsureTensorAtOutMemoryNode(
    ge::Graph* main,
    const std::string& tag,
    const EnsureOutMemoryInputs& inputs,
    fx_converter::Node* launch,
    std::vector<fx_converter::Node*>* kernel_nodes);

// Add a FreeBatchHbm node wired after Launch, with a control edge from launch.
fx_converter::Node* AddFreeBatchHbmNode(
    ge::Graph* main,
    const std::string& tag,
    const ge::TensorDesc& empty_td,
    uint64_t s_ws_addr,
    fx_converter::Node* alloc_ws,
    fx_converter::Node* launch,
    std::vector<fx_converter::Node*>* kernel_nodes);

// Fill the common ComputeNodeInfo fields of LoweringResult.
// Caller remains responsible for axis-specific fields (e.g. concat) and compute_nodes list.
void FillLoweringCniCommon(
    const OpCatalogEntry& entry,
    const std::vector<TensorRef>& inputs,
    const std::vector<int64_t>& out_shape,
    ge::DataType out_dtype,
    LoweringResult* out);

// Unpack Tensor* → (StorageShape*, GertTensorData*). Always creates a new Split node.
fx_converter::Node* CreateSplitDataTensor(
    LoweringContext& ctx,
    const TensorRef& ref,
    const std::string& tag,
    size_t index,
    uint64_t* shape_symbol,
    uint64_t* addr_symbol);

// Unpack Tensor* → (StorageShape*, GertTensorData*). Reuses an existing Split for the same packed value.
fx_converter::Node* GetOrCreateSplitDataTensor(
    LoweringContext& ctx,
    const TensorRef& ref,
    const std::string& tag,
    size_t index,
    uint64_t* shape_symbol,
    uint64_t* addr_symbol);

// GE: SplitDataTensor only for graph Data. Unpack every feed Tensor* once and write dual-value.
void UnpackGraphInputTensors(LoweringContext& ctx);

// Build a non-owning Tensor view from an already unpacked (shape, address) pair.
// This is the lazy boxed path used only by consumers whose ABI requires Tensor*.
fx_converter::Node* AddBuildRefTensorNode(
    ge::Graph* main,
    const std::string& tag,
    const ge::TensorDesc& td_o,
    const TensorEndpoint& shape,
    const TensorEndpoint& addr,
    ge::DataType dtype,
    uint64_t& sym_gen,
    uint64_t* s_out_tensor,
    std::vector<fx_converter::Node*>* kernel_nodes = nullptr);

// Populate TensorRef::producer/symbol on demand. Existing boxed refs are reused.
bool EnsureTensorRefBoxed(
    TensorRef* ref,
    ge::Graph* main,
    const std::string& tag,
    uint64_t& sym_gen,
    std::vector<fx_converter::Node*>* kernel_nodes = nullptr);

// Keep host layout payloads alive and refresh symbolic entries before each Execute.
bool AddRuntimeLayoutTensor(
    ge::Graph* main,
    ::BuiltExecuteGraph* built,
    const std::string& tag,
    const std::vector<int64_t>& values,
    const std::vector<fxrt::ir::SymbolicExprPtr>& expressions,
    uint64_t& sym_gen,
    std::vector<fx_converter::Node*>* kernel_nodes,
    TensorEndpoint* endpoint);

// Box a non-contiguous TensorRef for a wrapper that explicitly supports TensorV2.
bool EnsureStridedTensorRefBoxed(
    TensorRef* ref,
    ge::Graph* main,
    ::BuiltExecuteGraph* built,
    const std::string& tag,
    uint64_t& sym_gen,
    std::vector<fx_converter::Node*>* kernel_nodes = nullptr);

} // namespace fx_converter
