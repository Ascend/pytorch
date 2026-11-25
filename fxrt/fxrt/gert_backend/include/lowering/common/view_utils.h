#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "exe_graph/graph_builder_utils.h"
#include "lowering/op_lowering.h"

struct BuiltExecuteGraph;

namespace fx_converter {

// Compute contiguous row-major strides for a given size (product of dims from right).
// Dim -1 (dynamic/keep-dim) is allowed: trailing concrete dims still get strides; dims
// to the left of -1 get stride -1. Returns false if any dim < -1.
bool ContiguousRowMajorStrides(const std::vector<int64_t>& size, std::vector<int64_t>* strides);

// Check whether an existing stride matches contiguous row-major layout for the given size.
// size==-1 (dynamic) is a wildcard: trailing concrete dims still prove the stride product;
// a dim left of a dynamic extent cannot be proved. size==1 dims are stride-exempt.
// Returns false if ranks differ, any dim < -1, any stride < 0, or a non-unit stride mismatch.
bool IsContiguousRowMajor(const std::vector<int64_t>& size, const std::vector<int64_t>& stride);

// Returns false for an incompatible shape or an unproved symbolic layout.
bool ComputeReshapeViewStrides(
    const std::vector<int64_t>& input_size,
    const std::vector<int64_t>& input_stride,
    const std::vector<int64_t>& output_size,
    std::vector<int64_t>* output_stride);

// Shared layout kinds. Python LayoutKind in fx_layout_legalize.py uses the same names.
// Python builds the FX view (size/stride/offset + optional _fxrt_layout_kind);
// C++ re-validates from packed as_strided layout and chooses ViewAlias vs Path C.
enum class LayoutKind : uint8_t {
  kUnknown = 0,
  kDense, // contiguous row-major, ViewAlias reshape-style
  kBroadcast, // validated expand geometry, ViewAlias keeps parent storage
  kPermute, // permutation of a contiguous parent, ViewAlias keeps parent storage
  kGenericStrided, // anything else, including offset!=0; Path C AsStrided / Slice
};

// Classify as_strided(out_size, out_stride, offset) relative to parent in_shape.
// offset!=0 is always kGenericStrided (ViewAlias cannot represent a storage origin).
LayoutKind ClassifyAsStridedLayout(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_size,
    const std::vector<int64_t>& out_stride,
    int64_t offset);

// True when a dim is a real broadcast (stride 0 and extent is not 1/0).
bool HasBroadcastStride(const std::vector<int64_t>& size, const std::vector<int64_t>& stride);

// True when (out_size, out_stride) is a validated expand of a contiguous or
// last-two-permuted parent described by in_shape. A zero stride by itself is
// not enough: arbitrary overlapping as_strided layouts must use Path C.
bool IsBroadcastOfContiguous(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_size,
    const std::vector<int64_t>& out_stride);

// True when out (size, stride) is a permutation of a contiguous row-major parent.
// Identity (already row-major) and broadcast (stride 0) return false.
// Negative sizes are wildcards. Dynamic last-two t() may export stride [1,-1].
bool IsPermuteOfContiguous(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& out_size,
    const std::vector<int64_t>& out_stride);

// Dense = empty stride (legacy) or row-major + offset 0.
bool IsDenseTensorRef(const TensorRef& ref);

const std::vector<int64_t>& PhysicalShape(const TensorRef& ref);
const std::vector<int64_t>& LogicalShape(const TensorRef& ref);

// True when logical layout is a last-two-dim swap of a contiguous physical tensor.
bool IsLastTwoDimsTransposed(const TensorRef& ref);

// Product of shape dims; returns -1 if any dim < 0.
int64_t NumelOfShape(const std::vector<int64_t>& shape);

// Build a concrete (static) StorageShape Const from dims.
// All dims must be >= 0; returns false otherwise. Buffer kept alive via binary_bindata_blobs.
bool AddConcreteStorageShapeConst(
    ge::Graph* main,
    BuiltExecuteGraph* built,
    const std::string& tag,
    const std::vector<int64_t>& dims,
    uint64_t& sym_gen,
    std::vector<fx_converter::Node*>* kernel_nodes,
    fx_converter::Node** out_node,
    uint64_t* out_sym,
    std::string* err);

// Resolve the input TensorRef from lowering context by looking up node->inputs[0]
// in name_of -> tensors map. Returns false if input is missing or unresolved.
bool ResolveInputTensor(const fxrt::ir::NodePtr& node, const LoweringContext& ctx, TensorRef* in, std::string* err);

// Same lookup for an arbitrary producer node rather than node->inputs[0].
bool ResolveProducerTensor(
    const fxrt::ir::NodePtr& producer,
    const LoweringContext& ctx,
    TensorRef* in,
    std::string* err);

// ABI0 packs an as_strided's whole layout into one int_list: [rank] + size + stride + [offset].
// `what` prefixes the error string so the caller stays recognizable in a Convert failure.
bool ParseAsStridedPackedLayout(
    const std::vector<int64_t>& packed,
    const char* what,
    std::vector<int64_t>* size,
    std::vector<int64_t>* stride,
    int64_t* offset,
    std::string* err);

// Read a static layout from either the ABI0 packed tuple or separate size/stride/offset inputs.
// Symbolic values are left to the Path C collector, which preserves their expressions.
bool TryReadAsStridedLayout(
    const fxrt::ir::NodePtr& node,
    std::vector<int64_t>* size,
    std::vector<int64_t>* stride,
    int64_t* offset);

// True when an as_strided_view must use the Path C data adapter. In addition to
// static non-contiguous geometry, this includes symbolic stride/storage_offset
// payloads that are refreshed immediately before execution.
bool IsAsStridedPathC(const fxrt::ir::NodePtr& node);

// True for the fxrt IR ops that are metadata-only views: the output describes the
// input's storage with a different shape / stride / offset and never copies it.
bool IsStorageSharingViewOp(fxrt::ops::Op op);

// Walk back over storage-sharing views to the node that owns the storage `node` sits in.
// Returns `node` itself when it already owns its storage.
// Python analogue: fx_layout_legalize._same_rank_storage_root (same-rank) / as_strided base walk.
fxrt::ir::NodePtr StorageRootOf(const fxrt::ir::NodePtr& node);

// Like StorageRootOf, but stop at a rank-changing view (unsqueeze / squeeze / reshape).
// as_strided ViewAlias uses this so a [B,8,1] expand parent is not collapsed to [B,8].
fxrt::ir::NodePtr StorageRootSameRankOf(const fxrt::ir::NodePtr& node);

// Data input for an as_strided: its size / stride / storage_offset are addressed against
// the *storage* the input sits in, not against the input's logical layout, so a view in
// between must be walked past. The converter materializes several of those views into
// real buffers (permute_view -> Transpose, expand -> BroadcastTo, non-contiguous
// as_strided -> Slice / AsStrided); applying the strides to such a buffer would apply the
// view a second time. Resolves to the storage root's TensorRef and reports it in *base.
bool ResolveAsStridedBaseTensor(
    const fxrt::ir::NodePtr& node,
    const LoweringContext& ctx,
    TensorRef* in,
    fxrt::ir::NodePtr* base,
    std::string* err);

} // namespace fx_converter
