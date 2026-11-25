#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "graph/types.h"
#include "ir/value/value.h"
#include "exe_graph/graph_builder_utils.h"

// Defined in execute_graph_builder.h (global); forward decl avoids the heavy include here.
struct BuiltExecuteGraph;

namespace fx_converter {

struct AiCoreOpWork;
struct LoweringContext;

// Materialize a host-owned shape-list parameter for Path C:
//   - Launch: CopyFlowLaunch embeds it in launch args or copies it asynchronously.
//     The parameter's meta shape is [rank]; FXRT refresh only updates host values.
//   - Tiling BuildRefTensor StorageShape:
//       broadcast_target=true  (Expand): dims = list values (Broadcast AutoTiling target).
//       broadcast_target=false (Tile):   dims = [rank] (vector of multiples; payload holds values).
// Keep-dim (-1) templates are resolved by the ES runner before Execute (broadcast_target only).
//
// Payload width baseline: the type the value has in the original graph. Torch size / stride /
// storage_offset / index lists are int64 and reach us as fxrt::ir Tuple<int> -> ToIntList()
// (std::vector<int64_t>), so kGraphIntListDtype is what a call site uses unless a selected TBE
// binary leaves it no choice -- e.g. TopKV2 ships k only as `3,2` (OpCatalogEntry::
// shape_list_as_int32). Such a narrowing is a deliberate, checked exception, never a default.
constexpr ge::DataType kGraphIntListDtype = ge::DT_INT64;

// Materialize a shape list (dims + optional per-axis symbolic expressions) as the dual-view
// Const pair above. force_vector keeps a length-1 INT32 list a vector (AsStrided offsets);
// defer_tiling_view parks the tiling view on TensorRef::pending_tiling_host_ptr for
// BindBroadcastTilingViewsToInferShape.
bool MaterializeShapeTensor(
    ge::Graph* main,
    BuiltExecuteGraph* built,
    const std::vector<int64_t>& dims,
    const std::vector<fxrt::ir::SymbolicExprPtr>* axis_exprs,
    const std::string& tag,
    size_t idx,
    uint64_t& sym_gen,
    std::vector<fx_converter::Node*>* kernel_nodes,
    bool broadcast_target,
    ge::DataType payload_dtype,
    TensorRef& out,
    std::string* err,
    bool force_vector = false,
    bool defer_tiling_view = false);

// Materialize IR Double/Int/Bool as a host parameter (Broadcast-friendly [1]).
// A symbolic scalar refresh updates host_payload_out; dev_payload_out is null
// because CopyFlowLaunch owns transport and its device storage lifetime.
bool MaterializeScalarTensor(
    ge::Graph* main,
    BuiltExecuteGraph* built,
    const fxrt::ir::Value& scalar_value,
    ge::DataType dtype,
    const std::string& tag,
    size_t idx,
    uint64_t& sym_gen,
    std::vector<fx_converter::Node*>* kernel_nodes,
    TensorRef& out,
    std::string* err,
    void** host_payload_out = nullptr,
    void** dev_payload_out = nullptr);

// Broadcast-target shape list (Expand): the broadcast target IS the output shape, and the
// InferShape node already produces it as a live StorageShape (FxExprInferShape evaluates the
// symbolic rule, and GE's InferShape kernel fills both shape halves via TransformAllOutputsShape).
// Reusing that holder drops the refreshable-Const detour: no StorageShape object, no 8B pointer
// Const, no StorageShapeFromPtr, no CEM anchor (InferShape is not a constant expression), and no
// host-side keep-dim (-1) resolve for the tiling target.
void BindBroadcastTilingViewsToInferShape(LoweringContext& ctx, AiCoreOpWork& work);

} // namespace fx_converter
