#include "lowering/common/view_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>

#include "exe_graph/execute_graph_builder.h"
#include "mrt_ge/mrt_node_utils.h"
#include "shape/shape_list_utils.h"
#include "exe_graph/runtime/storage_shape.h"

namespace fx_converter {

bool ContiguousRowMajorStrides(const std::vector<int64_t> &size, std::vector<int64_t> *strides) {
  if (strides == nullptr) {
    return false;
  }
  strides->assign(size.size(), 0);
  int64_t expected = 1;
  for (int64_t i = static_cast<int64_t>(size.size()) - 1; i >= 0; --i) {
    const size_t ui = static_cast<size_t>(i);
    if (size[ui] < -1) {
      return false;
    }
    if (expected < 0) {
      (*strides)[ui] = -1;
      continue;
    }
    (*strides)[ui] = expected;
    if (size[ui] < 0) {
      expected = -1;
      continue;
    }
    expected *= (size[ui] == 0) ? 1 : size[ui];
  }
  return true;
}

bool ComputeReshapeViewStrides(const std::vector<int64_t> &input_size, const std::vector<int64_t> &input_stride,
                               const std::vector<int64_t> &output_size, std::vector<int64_t> *output_stride) {
  if (output_stride == nullptr || (!input_stride.empty() && input_size.size() != input_stride.size())) {
    return false;
  }
  auto product = [](const std::vector<int64_t> &dims, int64_t *result) {
    if (result == nullptr) {
      return false;
    }
    int64_t value = 1;
    for (const int64_t dim : dims) {
      if (dim < 0) {
        return false;
      }
      if (dim != 0 && value > std::numeric_limits<int64_t>::max() / dim) {
        return false;
      }
      value *= dim;
    }
    *result = value;
    return true;
  };
  auto multiply = [](int64_t lhs, int64_t rhs, int64_t *result) {
    if (result == nullptr || lhs < 0 || rhs < 0 || (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
      return false;
    }
    *result = lhs * rhs;
    return true;
  };

  std::vector<int64_t> source_stride = input_stride;
  if (source_stride.empty() && !ContiguousRowMajorStrides(input_size, &source_stride)) {
    return false;
  }
  if (std::any_of(input_size.begin(), input_size.end(), [](int64_t dim) { return dim < -1; }) ||
      std::any_of(source_stride.begin(), source_stride.end(), [](int64_t stride) { return stride < -1; }) ||
      std::any_of(output_size.begin(), output_size.end(), [](int64_t dim) { return dim < -1; })) {
    return false;
  }
  // Dense symbolic layouts can be reshaped without knowing the dynamic extents.
  // Static layouts use the chunk algorithm below even when contiguous:
  // singleton strides are observable and ATen view normalizes them by chunk.
  const auto dynamic_dim = [](int64_t dim) { return dim < 0; };
  const bool dynamic = std::any_of(input_size.begin(), input_size.end(), dynamic_dim) ||
                       std::any_of(source_stride.begin(), source_stride.end(), dynamic_dim) ||
                       std::any_of(output_size.begin(), output_size.end(), dynamic_dim);
  if (dynamic && IsContiguousRowMajor(input_size, source_stride)) {
    int64_t input_numel = 0;
    int64_t output_numel = 0;
    if (product(input_size, &input_numel) && product(output_size, &output_numel) && input_numel != output_numel) {
      return false;
    }
    return ContiguousRowMajorStrides(output_size, output_stride);
  }

  // The remaining chunk analysis needs exact geometry. Do not turn an unknown
  // dynamic stride into a guessed alias; the planner must adapt or fail.
  if (std::any_of(input_size.begin(), input_size.end(), [](int64_t dim) { return dim < 0; }) ||
      std::any_of(source_stride.begin(), source_stride.end(), [](int64_t stride) { return stride < 0; }) ||
      std::any_of(output_size.begin(), output_size.end(), [](int64_t dim) { return dim < 0; })) {
    return false;
  }
  int64_t input_numel = 0;
  int64_t output_numel = 0;
  if (!product(input_size, &input_numel) || !product(output_size, &output_numel) || input_numel != output_numel) {
    return false;
  }
  if (input_numel == 0) {
    if (input_size == output_size) {
      *output_stride = std::move(source_stride);
      return true;
    }
    return ContiguousRowMajorStrides(output_size, output_stride);
  }
  if (input_size.empty()) {
    return ContiguousRowMajorStrides(output_size, output_stride);
  }

  output_stride->assign(output_size.size(), 0);
  int64_t view_dim = static_cast<int64_t>(output_size.size()) - 1;
  int64_t chunk_base_stride = source_stride.back();
  int64_t tensor_numel = 1;
  int64_t view_numel = 1;
  for (int64_t tensor_dim = static_cast<int64_t>(input_size.size()) - 1; tensor_dim >= 0; --tensor_dim) {
    if (!multiply(tensor_numel, input_size[static_cast<size_t>(tensor_dim)], &tensor_numel)) {
      return false;
    }
    int64_t chunk_span = 0;
    if (!multiply(tensor_numel, chunk_base_stride, &chunk_span)) {
      return false;
    }
    const bool chunk_boundary =
        tensor_dim == 0 ||
        (input_size[static_cast<size_t>(tensor_dim - 1)] != 1 &&
         source_stride[static_cast<size_t>(tensor_dim - 1)] != chunk_span);
    if (!chunk_boundary) {
      continue;
    }

    while (view_dim >= 0 &&
           (view_numel < tensor_numel || output_size[static_cast<size_t>(view_dim)] == 1)) {
      int64_t stride_value = 0;
      if (!multiply(view_numel, chunk_base_stride, &stride_value)) {
        return false;
      }
      (*output_stride)[static_cast<size_t>(view_dim)] = stride_value;
      if (!multiply(view_numel, output_size[static_cast<size_t>(view_dim)], &view_numel)) {
        return false;
      }
      --view_dim;
    }
    if (view_numel != tensor_numel) {
      return false;
    }
    if (tensor_dim > 0) {
      chunk_base_stride = source_stride[static_cast<size_t>(tensor_dim - 1)];
      tensor_numel = 1;
      view_numel = 1;
    }
  }
  return view_dim == -1;
}

bool IsContiguousRowMajor(const std::vector<int64_t> &size, const std::vector<int64_t> &stride) {
  if (size.size() != stride.size()) {
    return false;
  }
  int64_t expected = 1;
  for (int64_t i = static_cast<int64_t>(size.size()) - 1; i >= 0; --i) {
    const size_t ui = static_cast<size_t>(i);
    if (size[ui] < -1 || stride[ui] < 0) {
      return false;
    }
    if (size[ui] == 0) {
      return true;
    }
    if (size[ui] == 1) {
      continue;
    }
    // Non-unit: static size>1 or dynamic size==-1. expected<0 means a dynamic dim
    // already appeared to the right, so this stride cannot be proved.
    if (expected < 0 || stride[ui] != expected) {
      return false;
    }
    expected = (size[ui] < 0) ? -1 : expected * size[ui];
  }
  return true;
}

bool HasBroadcastStride(const std::vector<int64_t> &size, const std::vector<int64_t> &stride) {
  if (size.size() != stride.size()) {
    return false;
  }
  for (size_t i = 0; i < size.size(); ++i) {
    if (stride[i] == 0 && size[i] != 1 && size[i] != 0) {
      return true;
    }
  }
  return false;
}

namespace {

struct BroadcastMap {
  std::vector<int64_t> mapped_size;
  std::vector<int64_t> mapped_stride;
  bool expanded = false;
  bool expanded_mapped_dim = false;
};

bool MapAlignedBroadcastAxis(int64_t in_dim, int64_t out_dim, int64_t stride, BroadcastMap *mapped) {
  if (stride == 0) {
    // Dynamic input extents cannot prove that a zero-stride dimension is an
    // expand of a singleton. Keep such layouts on Path C.
    if (in_dim != 1) {
      return false;
    }
    if (out_dim != 1) {
      mapped->expanded = true;
      mapped->expanded_mapped_dim = true;
    }
  } else {
    const bool dims_compatible = in_dim < 0 || out_dim < 0 || in_dim == out_dim;
    if (!dims_compatible || (in_dim == 1 && out_dim != 1) || stride < 0) {
      return false;
    }
  }
  mapped->mapped_size.push_back(out_dim);
  mapped->mapped_stride.push_back(stride);
  return true;
}

bool CollectBroadcastMappedAxes(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &out_size,
                                const std::vector<int64_t> &out_stride, BroadcastMap *mapped) {
  const size_t rank_delta = out_size.size() - in_shape.size();
  mapped->mapped_size.reserve(in_shape.size());
  mapped->mapped_stride.reserve(in_shape.size());
  for (size_t i = 0; i < out_size.size(); ++i) {
    if (out_size[i] < -1 || out_stride[i] < -1) {
      return false;
    }
    if (i < rank_delta) {
      if (out_stride[i] != 0) {
        return false;
      }
      mapped->expanded = true;
      continue;
    }
    const size_t j = i - rank_delta;
    if (!MapAlignedBroadcastAxis(in_shape[j], out_size[i], out_stride[i], mapped)) {
      return false;
    }
  }
  return true;
}

bool BroadcastMappedStridesExact(const BroadcastMap &mapped, const std::vector<int64_t> &in_contig) {
  for (size_t i = 0; i < mapped.mapped_size.size(); ++i) {
    if (mapped.mapped_size[i] == 1 || mapped.mapped_stride[i] == 0 || in_contig[i] < 0) {
      continue;
    }
    if (mapped.mapped_stride[i] != in_contig[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool IsBroadcastOfContiguous(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &out_size,
                             const std::vector<int64_t> &out_stride) {
  if (out_size.size() != out_stride.size() || out_size.size() < in_shape.size()) {
    return false;
  }
  std::vector<int64_t> in_contig;
  if (!ContiguousRowMajorStrides(in_shape, &in_contig)) {
    return false;
  }
  BroadcastMap mapped;
  if (!CollectBroadcastMappedAxes(in_shape, out_size, out_stride, &mapped)) {
    return false;
  }
  if (!mapped.expanded) {
    return false;
  }
  if (BroadcastMappedStridesExact(mapped, in_contig)) {
    return true;
  }
  // A broadcast over a transposed matrix (e.g. W.t().expand(B,N,N)) is still
  // valid; singleton expansion combined with a permutation is kept
  // conservative and will be materialized by Path C.
  return !mapped.expanded_mapped_dim &&
         IsPermuteOfContiguous(in_shape, mapped.mapped_size, mapped.mapped_stride);
}

LayoutKind ClassifyAsStridedLayout(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &out_size,
                                   const std::vector<int64_t> &out_stride, int64_t offset) {
  // ViewAlias has no storage-origin field. Non-zero offset is always Path C.
  if (offset != 0) {
    return LayoutKind::kGenericStrided;
  }
  if (IsBroadcastOfContiguous(in_shape, out_size, out_stride)) {
    return LayoutKind::kBroadcast;
  }
  if (IsPermuteOfContiguous(in_shape, out_size, out_stride)) {
    return LayoutKind::kPermute;
  }
  if (IsContiguousRowMajor(out_size, out_stride)) {
    return LayoutKind::kDense;
  }
  return LayoutKind::kGenericStrided;
}

namespace {

bool DimSizeCompatible(int64_t lhs, int64_t rhs) { return lhs < 0 || rhs < 0 || lhs == rhs; }

bool MatchPermuteDim(int64_t out_size, int64_t out_stride, int64_t in_size, int64_t in_stride) {
  if (!DimSizeCompatible(out_size, in_size)) {
    return false;
  }
  if (out_size == 1 || in_size == 1) {
    return out_size == 1 && in_size == 1;
  }
  if (out_stride < 0 || in_stride < 0) {
    return false;
  }
  return out_stride == in_stride;
}

// Dynamic 2D/last-two t(): export packs stride as [..., 1, -1] because the parent
// row stride is a SymInt. MatchPermuteDim rejects in_stride<0, so the general
// matcher cannot prove a permute. This is not the [-1,K] stride [K,1] identity
// (that is row-major / out_stride[-2]!=1).
bool IsDynamicLastTwoTranspose(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &out_size,
                               const std::vector<int64_t> &out_stride) {
  if (in_shape.size() < 2U || in_shape.size() != out_size.size() || out_size.size() != out_stride.size()) {
    return false;
  }
  const size_t r2 = in_shape.size() - 2U;
  const size_t r1 = in_shape.size() - 1U;
  if (out_stride[r2] != 1 || out_stride[r1] == 1 || out_stride[r1] == 0) {
    return false;
  }
  auto same_or_wild = [](int64_t a, int64_t b) { return a < 0 || b < 0 || a == b; };
  for (size_t i = 0; i < r2; ++i) {
    if (!same_or_wild(in_shape[i], out_size[i]) || out_stride[i] == 0) {
      return false;
    }
  }
  if (!same_or_wild(out_size[r2], in_shape[r1]) || !same_or_wild(out_size[r1], in_shape[r2])) {
    return false;
  }
  // Static parent extents pin the swapped row stride: t() of contiguous (..,H,W) is
  // (..,W,H)/(..,1,W). Only the SymInt wildcard (-1) may leave it open; any other
  // stride (e.g. an overlapping (1,2) read of a (2,4) buffer) is not a transpose.
  if (in_shape[r1] >= 0 && out_stride[r1] >= 0 && out_stride[r1] != in_shape[r1]) {
    return false;
  }
  return true;
}

// Greedy match of every out dim (size+stride) onto an unused in dim (size+contig stride).
// Size-1 dims only match size-1 in dims; the first definitive non-unit match pins its slot.
static bool TryMatchPermuteDims(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &out_size,
                                const std::vector<int64_t> &out_stride, const std::vector<int64_t> &in_contig) {
  std::vector<uint8_t> used(in_shape.size(), 0);
  std::vector<int> perm(out_size.size(), -1);
  for (size_t i = 0; i < out_size.size(); ++i) {
    int found = -1;
    for (size_t j = 0; j < in_shape.size(); ++j) {
      if (used[j] != 0) {
        continue;
      }
      if (!MatchPermuteDim(out_size[i], out_stride[i], in_shape[j], in_contig[j])) {
        continue;
      }
      found = static_cast<int>(j);
      if (out_size[i] != 1 && in_shape[j] != 1) {
        break;
      }
    }
    if (found < 0) {
      return false;
    }
    used[static_cast<size_t>(found)] = 1;
    perm[i] = found;
  }
  // Identity is row-major, not a permute. Dynamic leading dims (e.g. [-1, 4024]
  // stride [4024, 1]) used to fail IsContiguousRowMajor and fall through as a
  // false-positive permute; DualStage mm then set transpose_x1 and aclnnMm
  // reported K-axis mismatch (self Kdim[batch], mat2 Kdim[K]).
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int>(i)) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool IsPermuteOfContiguous(const std::vector<int64_t> &in_shape, const std::vector<int64_t> &out_size,
                           const std::vector<int64_t> &out_stride) {
  if (in_shape.empty() || in_shape.size() != out_size.size() || out_size.size() != out_stride.size()) {
    return false;
  }
  if (HasBroadcastStride(out_size, out_stride) || IsContiguousRowMajor(out_size, out_stride)) {
    return false;
  }
  if (IsDynamicLastTwoTranspose(in_shape, out_size, out_stride)) {
    return true;
  }
  std::vector<int64_t> in_contig;
  if (!ContiguousRowMajorStrides(in_shape, &in_contig)) {
    return false;
  }
  const int64_t in_n = NumelOfShape(in_shape);
  const int64_t out_n = NumelOfShape(out_size);
  if (in_n >= 0 && out_n >= 0 && in_n != out_n) {
    return false;
  }
  return TryMatchPermuteDims(in_shape, out_size, out_stride, in_contig);
}

bool IsDenseTensorRef(const TensorRef &ref) {
  if (ref.storage_offset != 0 || ref.storage_offset_expr != nullptr || ref.runtime_storage_offset.Valid()) {
    return false;
  }
  if (ref.proven_dense) {
    return true;
  }
  if (std::any_of(ref.stride_exprs.begin(), ref.stride_exprs.end(),
                  [](const auto &expr) { return expr != nullptr; })) return false;
  if (ref.stride.empty()) return true;
  // `ref.shape` is GE physical StorageShape (may be NPU-padded). Contiguity
  // is an ATen logical property and must use the logical size.
  return IsContiguousRowMajor(LogicalShape(ref), ref.stride);
}

const std::vector<int64_t> &PhysicalShape(const TensorRef &ref) {
  return ref.storage_shape.empty() ? ref.shape : ref.storage_shape;
}

const std::vector<int64_t> &LogicalShape(const TensorRef &ref) {
  return ref.logical_shape.empty() ? ref.shape : ref.logical_shape;
}

bool IsLastTwoDimsTransposed(const TensorRef &ref) {
  const std::vector<int64_t> &physical = PhysicalShape(ref);
  const std::vector<int64_t> &logical = LogicalShape(ref);
  if (physical.size() < 2U || physical.size() != logical.size() || ref.stride.size() != logical.size()) {
    return false;
  }
  // Dense row-major (including dynamic batch, e.g. reshape [s0,1,K]→[s0,K]) is
  // never a last-two swap. Wildcards make physical==logical look swap-compatible.
  if (IsContiguousRowMajor(logical, ref.stride)) {
    return false;
  }
  const size_t rank = physical.size();
  for (size_t i = 0; i + 2U < rank; ++i) {
    if (physical[i] >= 0 && logical[i] >= 0 && physical[i] != logical[i]) {
      return false;
    }
  }
  const size_t r2 = rank - 2U;
  const size_t r1 = rank - 1U;
  if (logical[r2] >= 0 && physical[r1] >= 0 && logical[r2] != physical[r1]) {
    return false;
  }
  if (logical[r1] >= 0 && physical[r2] >= 0 && logical[r1] != physical[r2]) {
    return false;
  }
  return IsPermuteOfContiguous(physical, logical, ref.stride);
}

int64_t NumelOfShape(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return 1;
  }
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d < 0) {
      return -1;
    }
    if (d != 0 && n > std::numeric_limits<int64_t>::max() / d) return -1;
    n *= d;
  }
  return n;
}

bool AddConcreteStorageShapeConst(ge::Graph *main, BuiltExecuteGraph *built, const std::string &tag,
                                  const std::vector<int64_t> &dims, uint64_t &sym_gen,
                                  std::vector<fx_converter::Node *> *kernel_nodes, fx_converter::Node **out_node,
                                  uint64_t *out_sym, std::string *err) {
  if (out_node == nullptr || out_sym == nullptr || built == nullptr) {
    return false;
  }
  if (std::any_of(dims.begin(), dims.end(), [](int64_t d) { return d < 0; })) {
    if (err != nullptr) {
      *err = "concrete StorageShape fallback needs static dims";
    }
    return false;
  }
  gert::StorageShape ss;
  ss.MutableStorageShape().SetDimNum(dims.size());
  ss.MutableOriginShape().SetDimNum(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    ss.MutableStorageShape().SetDim(static_cast<int64_t>(i), dims[i]);
    ss.MutableOriginShape().SetDim(static_cast<int64_t>(i), dims[i]);
  }
  auto ss_blob = std::make_shared<std::vector<uint8_t>>(sizeof(gert::StorageShape));
  std::memcpy(ss_blob->data(), &ss, sizeof(gert::StorageShape));
  built->binary_bindata_blobs.push_back(ss_blob);
  *out_node = AddConstNode(main, "const_storage_shape_" + tag, ss_blob->data(), ss_blob->size(), false, sym_gen,
                           FxrtConstType::kGertStorageShapePtr);
  if (kernel_nodes != nullptr) {
    kernel_nodes->push_back(*out_node);
  }
  *out_sym = (*out_node)->GetExtendInfo()->GetOutputSymbol(0);
  return true;
}

bool ResolveProducerTensor(const fxrt::ir::NodePtr &producer, const LoweringContext &ctx, TensorRef *in,
                           std::string *err) {
  if (producer == nullptr) {
    if (err != nullptr) {
      *err = "lowering: missing tensor input";
    }
    return false;
  }
  const auto name_it = ctx.name_of->find(producer.get());
  if (name_it == ctx.name_of->end()) {
    if (err != nullptr) {
      *err = "lowering: unnamed input producer";
    }
    return false;
  }
  const auto tens_it = ctx.tensors->find(name_it->second);
  if (tens_it == ctx.tensors->end()) {
    if (err != nullptr) {
      *err = "lowering: unresolved input tensor '" + name_it->second + "'";
    }
    return false;
  }
  *in = tens_it->second;
  return true;
}

bool ResolveInputTensor(const fxrt::ir::NodePtr &node, const LoweringContext &ctx, TensorRef *in, std::string *err) {
  if (node == nullptr || node->inputs.empty()) {
    if (err != nullptr) {
      *err = "lowering: missing tensor input";
    }
    return false;
  }
  return ResolveProducerTensor(node->inputs[0], ctx, in, err);
}

bool ParseAsStridedPackedLayout(const std::vector<int64_t> &packed, const char *what, std::vector<int64_t> *size,
                                std::vector<int64_t> *stride, int64_t *offset, std::string *err) {
  if (size == nullptr || stride == nullptr || offset == nullptr) {
    return false;
  }
  const std::string prefix = std::string(what == nullptr ? "as_strided" : what) + ": ";
  *offset = 0;
  if (packed.empty()) {
    if (err != nullptr) {
      *err = prefix + "as_strided missing packed int_list";
    }
    return false;
  }
  const int64_t rank = packed[0];
  if (rank < 0 || packed.size() < static_cast<size_t>(1 + rank)) {
    if (err != nullptr) {
      *err = prefix + "as_strided invalid rank in packed int_list";
    }
    return false;
  }
  size->assign(packed.begin() + 1, packed.begin() + 1 + rank);
  const size_t after_size = static_cast<size_t>(1 + rank);
  if (packed.size() < after_size + static_cast<size_t>(rank) + 1U) {
    if (err != nullptr) {
      *err = prefix + "as_strided packed int_list incomplete";
    }
    return false;
  }
  stride->assign(packed.begin() + static_cast<std::ptrdiff_t>(after_size),
                 packed.begin() + static_cast<std::ptrdiff_t>(after_size + static_cast<size_t>(rank)));
  *offset = packed[after_size + static_cast<size_t>(rank)];
  return true;
}

bool TryReadAsStridedLayout(const fxrt::ir::NodePtr &node, std::vector<int64_t> *size, std::vector<int64_t> *stride,
                            int64_t *offset) {
  if (node == nullptr || size == nullptr || stride == nullptr || offset == nullptr) {
    return false;
  }
  std::vector<fxrt::ir::NodePtr> lists;
  for (size_t i = 1; i < node->inputs.size(); ++i) {
    const auto &in = node->inputs[i];
    if (in == nullptr || in->output == nullptr || !in->output->IsTuple()) {
      continue;
    }
    const auto &tuple = in->output->ToTuple();
    if (std::any_of(tuple->begin(), tuple->end(), [](const fxrt::ir::ValuePtr &value) {
          return value != nullptr && value->IsSymbol();
        })) {
      return false;
    }
    lists.push_back(in);
  }
  if (lists.size() == 1U) {
    const auto packed = ReadIntListSymbolSafe(lists[0]->output->ToTuple());
    if (packed.empty() || packed[0] < 0 ||
        packed.size() != static_cast<size_t>(1 + 2 * packed[0] + 1)) {
      return false;
    }
    std::string err;
    return ParseAsStridedPackedLayout(packed, "as_strided", size, stride, offset, &err);
  }
  if (lists.size() != 2U) {
    return false;
  }
  *size = ReadIntListSymbolSafe(lists[0]->output->ToTuple());
  *stride = ReadIntListSymbolSafe(lists[1]->output->ToTuple());
  if (size->size() != stride->size()) {
    return false;
  }
  *offset = 0;
  for (size_t i = 1; i < node->inputs.size(); ++i) {
    const auto &in = node->inputs[i];
    if (in == nullptr || in->output == nullptr || in->output->IsTuple() || in->output->IsNone()) {
      continue;
    }
    if (in->output->IsSymbol()) {
      return false;
    }
    if (in->output->IsInt()) {
      *offset = in->output->ToInt();
      return true;
    }
  }
  return true;
}

bool IsStorageSharingViewOp(fxrt::ops::Op op) {
  switch (op) {
    case fxrt::ops::Op_view:
    case fxrt::ops::Op_reshape:
    case fxrt::ops::Op_alias:
    case fxrt::ops::Op_expand:
    case fxrt::ops::Op_squeeze:
    case fxrt::ops::Op_squeeze_dim:
    case fxrt::ops::Op_unsqueeze:
    case fxrt::ops::Op_as_strided_view:
    case fxrt::ops::Op_flatten_view:
    case fxrt::ops::Op_narrow_view:
    case fxrt::ops::Op_permute_view:
    case fxrt::ops::Op_select_view:
    case fxrt::ops::Op_slice_view:
    case fxrt::ops::Op_squeeze_view:
    case fxrt::ops::Op_unsqueeze_view:
      return true;
    default:
      return false;
  }
}

namespace {

fxrt::ir::NodePtr WalkStorageRoot(const fxrt::ir::NodePtr &node, bool same_rank_only) {
  constexpr size_t kMaxViewChainDepth = 64;
  fxrt::ir::NodePtr cur = node;
  for (size_t depth = 0; depth < kMaxViewChainDepth; ++depth) {
    if (cur == nullptr || !IsStorageSharingViewOp(cur->op) || cur->inputs.empty()) {
      break;
    }
    const auto &parent = cur->inputs[0];
    if (parent == nullptr || parent->output == nullptr || !parent->output->IsTensor()) {
      break;
    }
    if (same_rank_only && ShapeOfNode(parent).size() != ShapeOfNode(cur).size()) {
      break;
    }
    cur = parent;
  }
  return cur;
}

}  // namespace

fxrt::ir::NodePtr StorageRootOf(const fxrt::ir::NodePtr &node) { return WalkStorageRoot(node, false); }

fxrt::ir::NodePtr StorageRootSameRankOf(const fxrt::ir::NodePtr &node) { return WalkStorageRoot(node, true); }

bool ResolveAsStridedBaseTensor(const fxrt::ir::NodePtr &node, const LoweringContext &ctx, TensorRef *in,
                                fxrt::ir::NodePtr *base, std::string *err) {
  if (node == nullptr || node->inputs.empty()) {
    if (err != nullptr) {
      *err = "lowering: missing tensor input";
    }
    return false;
  }
  // The planner has already chosen the storage address, capacity shape and
  // matching geometry. Walking the original IR again can select a logical
  // subview header or undo address rebasing while retaining its packed offset.
  if (node->attrs.count("fxrt_write_dense") != 0U) {
    if (base != nullptr) *base = node->inputs[0];
    return ResolveInputTensor(node, ctx, in, err);
  }
  // A reshape may have acquired a new allocation through the edge planner. Do
  // not walk past that allocation just because the IR op is called a view.
  fxrt::ir::NodePtr root = node->inputs[0];
  for (size_t depth = 0; depth < 64U && root != nullptr && IsStorageSharingViewOp(root->op) &&
                         !root->inputs.empty(); ++depth) {
    const auto &parent = root->inputs[0];
    if (parent == nullptr || parent->output == nullptr || !parent->output->IsTensor() ||
        ShapeOfNode(parent).size() != ShapeOfNode(root).size()) break;
    TensorRef current_ref;
    TensorRef parent_ref;
    std::string ignored;
    if (!ResolveProducerTensor(root, ctx, &current_ref, &ignored) ||
        !ResolveProducerTensor(parent, ctx, &parent_ref, &ignored) ||
        !current_ref.unpacked.addr.Valid() || !parent_ref.unpacked.addr.Valid() ||
        current_ref.unpacked.addr.node != parent_ref.unpacked.addr.node ||
        current_ref.unpacked.addr.out != parent_ref.unpacked.addr.out) break;
    root = parent;
  }
  if (base != nullptr) {
    *base = root;
  }
  return ResolveProducerTensor(root, ctx, in, err);
}

}  // namespace fx_converter
