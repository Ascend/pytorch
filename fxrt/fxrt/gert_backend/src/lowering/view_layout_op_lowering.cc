#include <algorithm>
#include <limits>
#include <sstream>
#include <vector>

#include "common/logger.h"
#include "lowering/common/view_utils.h"
#include "lowering/common/storage_geometry.h"
#include "lowering/common/view_metadata.h"
#include "lowering/common/wiring.h"
#include "lowering/op_lowering_registry.h"
#include "mrt_ge/mrt_node_utils.h"
#include "ops/op_def/ops_name.h"
#include "shape/fx_sym_expr_parser.h"
#include "shape/shape_list_utils.h"

namespace fx_converter {
namespace {
std::string LayoutValues(const std::vector<int64_t> &values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0U) stream << ",";
    stream << values[i];
  }
  stream << "]";
  return stream.str();
}

struct ViewGeometry {
  TensorRef source;
  std::vector<int64_t> size;
  std::vector<fxrt::ir::SymbolicExprPtr> size_exprs;
  std::vector<int64_t> stride;
  std::vector<fxrt::ir::SymbolicExprPtr> stride_exprs;
  int64_t offset = 0;
  fxrt::ir::SymbolicExprPtr offset_expr;
  TensorEndpoint runtime_offset;
  bool native_layout = false;
  bool aliases = true;
  bool contiguous = false;
};

bool ReadAsStridedGeometry(const fxrt::ir::NodePtr &node, FxSymbolEnv *env, ViewGeometry *view) {
  if (node == nullptr || env == nullptr || view == nullptr) return false;
  std::vector<fxrt::ir::NodePtr> lists;
  std::vector<fxrt::ir::NodePtr> scalars;
  for (size_t i = 1; i < node->inputs.size(); ++i) {
    const auto &input = node->inputs[i];
    if (input == nullptr || input->output == nullptr || input->output->IsNone()) continue;
    if (input->output->IsTuple()) lists.push_back(input);
    else scalars.push_back(input);
  }
  if (lists.empty() || lists.size() > 2U || scalars.size() > 1U) return false;
  if (lists.size() == 1U) {
    std::vector<int64_t> packed;
    std::vector<fxrt::ir::SymbolicExprPtr> exprs;
    if (!ReadIntListWithSymbols(lists[0]->output->ToTuple(), *env, &packed, &exprs) || packed.empty() ||
        packed[0] < 0 || packed.size() != static_cast<size_t>(2 + 2 * packed[0]) || exprs.size() != packed.size()) {
      return false;
    }
    const size_t rank = static_cast<size_t>(packed[0]);
    view->size.assign(packed.begin() + 1, packed.begin() + 1 + static_cast<std::ptrdiff_t>(rank));
    view->size_exprs.assign(exprs.begin() + 1, exprs.begin() + 1 + static_cast<std::ptrdiff_t>(rank));
    view->stride.assign(packed.begin() + 1 + static_cast<std::ptrdiff_t>(rank),
                        packed.begin() + 1 + static_cast<std::ptrdiff_t>(2U * rank));
    view->stride_exprs.assign(exprs.begin() + 1 + static_cast<std::ptrdiff_t>(rank),
                              exprs.begin() + 1 + static_cast<std::ptrdiff_t>(2U * rank));
    view->offset = packed.back();
    view->offset_expr = exprs.back();
    return true;
  }
  if (!ReadIntListWithSymbols(lists[0]->output->ToTuple(), *env, &view->size, &view->size_exprs) ||
      !ReadIntListWithSymbols(lists[1]->output->ToTuple(), *env, &view->stride, &view->stride_exprs) ||
      view->size.size() != view->stride.size()) return false;
  if (scalars.empty()) return true;
  const auto &scalar = scalars.front()->output;
  if (scalar->IsInt()) {
    view->offset = scalar->ToInt();
    return true;
  }
  if (scalar->IsSymbol()) {
    view->offset_expr = ParseFxSymExpr(scalar->ToSymbol(), *env);
    return view->offset_expr != nullptr;
  }
  return false;
}

void SetStorageGeometry(const StorageGeometry &storage, ViewGeometry *view) {
  view->source = storage.anchor;
  view->offset = storage.offset;
  view->offset_expr = storage.offset_expr;
  view->runtime_offset = storage.runtime_offset;
}

// Derive static geometry from the actual lowered allocation. Preserve both
// contiguous and strided views; only the planner chooses address adaptation or
// a device conversion. Unknown geometry keeps its existing implementation.
bool ResolveGeometry(const fxrt::ir::NodePtr &node, const LoweringContext &ctx, ViewGeometry *geometry) {
  if (node == nullptr || geometry == nullptr || node->attrs.count("fxrt_write_dense") != 0U) return false;
  auto &view = *geometry;
  NativeViewLayout native;
  if (ResolveInputTensor(node, ctx, &view.source, nullptr) &&
      ReadNativeViewLayout(node, view.source, ctx.symbol_env, &native)) {
    view.size = native.size;
    view.size_exprs = native.size_exprs;
    view.stride = native.stride;
    view.stride_exprs = native.stride_exprs;
    StorageGeometry storage;
    if (!ComposeViewStorageGeometry(view.source, ViewOffsetSemantics::kImmediateRelative,
                                     native.relative_offset, native.relative_offset_expr, &storage)) return false;
    SetStorageGeometry(storage, &view);
    view.native_layout = true;
    view.aliases = native.aliases;
    view.contiguous = native.contiguous;
    return true;
  }
  if (node->attrs.count("fxrt_native_view_layouts") != 0U) return false;
  const auto slice = node->attrs.find("fxrt_slice_geometry");
  const auto select = node->attrs.find("fxrt_select_geometry");
  if (slice != node->attrs.end() || select != node->attrs.end()) {
    if (!ResolveInputTensor(node, ctx, &view.source, nullptr)) return false;
    view.size = LogicalShape(view.source);
    view.stride = view.source.stride;
    if (view.stride.empty() && !ContiguousRowMajorStrides(view.size, &view.stride)) return false;
    if (view.size.size() != view.stride.size()) return false;
    view.offset = 0;
    const auto attribute = slice != node->attrs.end() ? slice->second : select->second;
    if (attribute == nullptr || !attribute->IsTuple()) return false;
    const auto args = ReadIntListSymbolSafe(attribute->ToTuple());
    if (slice != node->attrs.end()) {
      if (args.size() % 4U != 0U) return false;
      for (size_t i = 0; i < args.size(); i += 4U) {
        const auto axis = args[i];
        if (axis < 0 || static_cast<size_t>(axis) >= view.size.size() || args[i + 1] < 0 ||
            args[i + 2] < 0 || args[i + 3] <= 0 || view.stride[axis] < 0) return false;
        view.offset += args[i + 1] * view.stride[axis];
        view.stride[axis] *= args[i + 3];
        view.size[axis] = args[i + 2];
      }
    } else {
      if (args.size() != 2U || args[0] < 0 || static_cast<size_t>(args[0]) >= view.size.size() ||
          args[1] < 0 || args[1] >= view.size[args[0]] || view.stride[args[0]] < 0) return false;
      view.offset += args[1] * view.stride[args[0]];
      view.size.erase(view.size.begin() + args[0]);
      view.stride.erase(view.stride.begin() + args[0]);
    }
  } else if (node->op == fxrt::ops::Op_as_strided_view) {
    if (!ReadAsStridedGeometry(node, ctx.symbol_env, &view)) return false;
    ViewOffsetSemantics semantics;
    if (!ReadViewOffsetSemantics(node, &semantics)) return false;
    if (node->attrs.count("fxrt_as_strided_offset_semantics") == 0U &&
        view.offset == -1 && view.offset_expr == nullptr) semantics = ViewOffsetSemantics::kInherit;
    StorageGeometry storage;
    if (!ResolveViewStorageGeometry(node, ctx, semantics, view.offset, view.offset_expr, &storage)) return false;
    SetStorageGeometry(storage, &view);
  } else if (node->op == fxrt::ops::Op_permute_view || node->op == fxrt::ops::Op_expand) {
    if (node->inputs.size() != 2U || !ResolveInputTensor(node, ctx, &view.source, nullptr) ||
        node->inputs[1]->output == nullptr || !node->inputs[1]->output->IsTuple()) return false;
    const auto argument = ReadIntListSymbolSafe(node->inputs[1]->output->ToTuple());
    const auto &input_size = LogicalShape(view.source);
    auto input_stride = view.source.stride;
    if (input_stride.empty() && !ContiguousRowMajorStrides(input_size, &input_stride)) return false;
    if (input_stride.size() != input_size.size()) return false;
    view.offset = 0;
    if (node->op == fxrt::ops::Op_permute_view) {
      if (argument.size() != input_size.size()) return false;
      std::vector<bool> used(input_size.size(), false);
      for (int64_t axis : argument) {
        if (axis < 0) axis += static_cast<int64_t>(input_size.size());
        if (axis < 0 || static_cast<size_t>(axis) >= used.size() || used[axis]) return false;
        used[axis] = true;
        view.size.push_back(input_size[axis]);
        view.stride.push_back(input_stride[axis]);
      }
    } else {
      if (argument.size() < input_size.size()) return false;
      const size_t leading = argument.size() - input_size.size();
      for (size_t i = 0; i < argument.size(); ++i) {
        const int64_t original = i < leading ? 1 : input_size[i - leading];
        const int64_t size = argument[i] == -1 && i >= leading ? original : argument[i];
        if (size < 0 || original < 0 || (original != 1 && original != size)) return false;
        view.size.push_back(size);
        view.stride.push_back(i < leading || original != size ? 0 : input_stride[i - leading]);
      }
    }
  } else if (node->op == fxrt::ops::Op_slice || node->op == fxrt::ops::Op_getitem_slice) {
    if (node->inputs.size() != 3U || !ResolveInputTensor(node, ctx, &view.source, nullptr)) return false;
    for (size_t i = 1U; i < 3U; ++i) {
      if (node->inputs[i] == nullptr || node->inputs[i]->output == nullptr ||
          !node->inputs[i]->output->IsTuple()) return false;
    }
    const auto starts = ReadIntListSymbolSafe(node->inputs[1]->output->ToTuple());
    view.size = ReadIntListSymbolSafe(node->inputs[2]->output->ToTuple());
    const auto &input_size = LogicalShape(view.source);
    view.stride = view.source.stride;
    if (view.stride.empty() && !ContiguousRowMajorStrides(input_size, &view.stride)) return false;
    if (starts.size() != input_size.size() || view.size.size() != input_size.size() ||
        view.stride.size() != input_size.size()) return false;
    view.offset = 0;
    for (size_t i = 0U; i < starts.size(); ++i) {
      if (input_size[i] < 0 || starts[i] < 0 || view.size[i] < 0 || view.stride[i] < 0 ||
          starts[i] > input_size[i] || view.size[i] > input_size[i] - starts[i] || view.offset < 0 ||
          (starts[i] != 0 && view.stride[i] > (std::numeric_limits<int64_t>::max() - view.offset) / starts[i])) {
        return false;
      }
      view.offset += starts[i] * view.stride[i];
    }
  } else {
    return false;
  }
  if (node->op != fxrt::ops::Op_as_strided_view) {
    StorageGeometry storage;
    if (!ComposeViewStorageGeometry(view.source, ViewOffsetSemantics::kImmediateRelative,
                                     view.offset, view.offset_expr, &storage)) return false;
    SetStorageGeometry(storage, &view);
  }
  if (view.offset < 0 && view.offset_expr == nullptr) return false;
  if (view.size.size() != view.stride.size()) return false;
  if (view.size_exprs.empty()) view.size_exprs.resize(view.size.size());
  if (view.stride_exprs.empty()) view.stride_exprs.resize(view.stride.size());
  if (view.size_exprs.size() != view.size.size() || view.stride_exprs.size() != view.stride.size()) return false;
  for (size_t i = 0; i < view.size.size(); ++i) {
    if (view.size[i] < 0 && view.size_exprs[i] == nullptr) return false;
    if (view.stride[i] < 0 && view.stride_exprs[i] == nullptr) return false;
  }
  if (std::any_of(view.size.begin(), view.size.end(), [](int64_t dim) { return dim < 0; }) ||
      std::any_of(view.stride.begin(), view.stride.end(), [](int64_t stride) { return stride < 0; }) ||
      view.offset_expr != nullptr) {
    return true;
  }
  // External storage capacity is a property of the current Execute, not the
  // feed's logical shape or the tracing example. A public absolute view may
  // legitimately reach beyond that shape within the same backing allocation.
  const bool runtime_capacity = view.source.feed_index >= 0;
  const int64_t capacity = runtime_capacity ? -1 : NumelOfShape(PhysicalShape(view.source));
  const int64_t count = NumelOfShape(view.size);
  if (capacity < 0 && !runtime_capacity && !IsContiguousRowMajor(view.size, view.stride)) return false;
  int64_t end = view.offset;
  for (size_t i = 0; i < view.size.size(); ++i) {
    const int64_t steps = std::max<int64_t>(view.size[i] - 1, 0);
    if (steps != 0 && view.stride[i] > (std::numeric_limits<int64_t>::max() - end) / steps) return false;
    end += steps * view.stride[i];
  }
  // External and symbolic capacities retain their runtime bounds check at
  // address adaptation. Internal allocations keep the static capacity check.
  return count >= 0 && (capacity < 0 || (count == 0 ? view.offset <= capacity : end < capacity));
}

class ViewLayoutOpLowering final : public OpLowering {
 public:
  const char *Name() const override { return "ViewLayout"; }

  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *ctx) const override {
    if (ctx == nullptr) return false;  // Catalog/ViewAlias performs the support precheck.
    if (node->op == fxrt::ops::Op_as_strided_view &&
        node->attrs.count("fxrt_as_strided_offset_semantics") != 0U &&
        node->attrs.count("fxrt_write_dense") == 0U) {
      // Once the entry declared offset semantics, an unsupported geometry
      // must fail here. Older fallback lowerings only understand absolute
      // arguments and must not silently reinterpret an unresolved relative view.
      return true;
    }
    if (node->attrs.count("fxrt_native_view_layouts") != 0U && node->attrs.count("fxrt_write_dense") == 0U) {
      return true;
    }
    ViewGeometry geometry;
    return ResolveGeometry(node, *ctx, &geometry);
  }

  InputLayoutKind InputLayout(const fxrt::ir::NodePtr &node, size_t) const override {
    // Geometry is relative to the original storage. Do not rebase it on input.
    return node->attrs.count("fxrt_native_view_layouts") != 0U
               ? InputLayoutKind::kMetadata : InputLayoutKind::kStorageIndexed;
  }

  bool Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx,
             LoweringResult *out, std::string *err) override {
    if (out == nullptr || !ValidateLoweringContext(ctx, false, Name(), err)) return false;
    ViewGeometry geometry;
    if (!ResolveGeometry(node, ctx, &geometry)) {
      if (err != nullptr) *err = "view storage geometry cannot be resolved against the actual producer";
      return false;
    }
    if (!geometry.aliases) {
      if (err != nullptr) *err = "native view: copying reshape requires a dense planner input";
      return false;
    }
    auto &source = geometry.source;
    TensorEndpoint address = source.unpacked.addr;
    if (!address.Valid()) {
      auto *split = GetOrCreateSplitDataTensor(ctx, source, "view_" + tag, 0, nullptr, &address.sym);
      address.node = split;
      address.out = 1;
    }
    if (!address.Valid()) return false;
    TensorRef result;
    result.dtype = source.dtype;
    result.shape = geometry.size;
    result.stride = geometry.stride;
    result.logical_shape_exprs = geometry.size_exprs;
    result.stride_exprs = geometry.stride_exprs;
    result.storage_shape = PhysicalShape(source);
    result.storage_offset = geometry.offset;
    result.storage_offset_expr = geometry.offset_expr;
    result.feed_index = source.feed_index;
    result.runtime_storage_offset = geometry.runtime_offset;
    result.proven_dense = geometry.native_layout ? geometry.contiguous
                            : IsContiguousRowMajor(geometry.size, geometry.stride);
    result.unpacked.addr = address;
    result.allocation = source.allocation.Valid() ? source.allocation : address;
    const bool dynamic_shape = std::any_of(result.shape.begin(), result.shape.end(), [](int64_t dim) { return dim < 0; });
    if (dynamic_shape) {
      TensorEndpoint layout;
      if (!AddRuntimeLayoutTensor(ctx.main, ctx.built, tag + "_shape", result.shape,
                                  result.logical_shape_exprs, *ctx.sym_gen, ctx.kernel_nodes, &layout)) {
        if (err != nullptr) *err = "native view: cannot bind logical shape expressions";
        return false;
      }
      const ge::TensorDesc empty;
      RtGraphBuilder builder(ctx.main, empty, *ctx.sym_gen, ctx.kernel_nodes);
      TensorEndpoint runtime_shape = source.unpacked.shape;
      if (!runtime_shape.Valid()) {
        auto *split = GetOrCreateSplitDataTensor(ctx, source, "view_shape_input_" + tag, 0,
                                                 &runtime_shape.sym, nullptr);
        runtime_shape = {split, 0, runtime_shape.sym};
      }
      if (!runtime_shape.Valid()) return false;
      result.unpacked.shape =
          builder.Node("FxBuildLayoutShape", "view_shape_" + tag, {layout, runtime_shape}, 1U).outs[0];
      // The host payload is refreshed before each Execute. Its Const address
      // must not make GE hoist the shape read into the model's init graph.
      if (ctx.bind_symbols == nullptr) {
        if (err != nullptr) *err = "native view: symbolic layout has no runtime symbol binding";
        return false;
      }
    } else if (!AddConcreteStorageShapeConst(ctx.main, ctx.built, tag + "_view_shape", result.shape,
                                              *ctx.sym_gen, ctx.kernel_nodes, &result.unpacked.shape.node,
                                              &result.unpacked.shape.sym, err)) {
      return false;
    }
    if (!result.proven_dense) {
      result.logical_shape = geometry.size;
      result.logical_shape_ep = result.unpacked.shape;
      result.shape = PhysicalShape(source);
      result.unpacked.shape = source.unpacked.shape;
      if (!result.unpacked.shape.Valid()) {
        auto *split = GetOrCreateSplitDataTensor(ctx, source, "view_base_" + tag, 0,
                                               &result.unpacked.shape.sym, nullptr);
        result.unpacked.shape.node = split;
        result.unpacked.shape.out = 0;
      }
    }
    // Keep the canonical storage anchor for subsequent absolute views. Dense
    // consumers rebase through EdgeLayoutPlanner; metadata views never consume
    // the offset or shrink their storage capacity as a side effect.

    out->outputs = {std::move(result)};
    out->skip_cni = true;
    return true;
  }

};
}  // namespace
}  // namespace fx_converter

// Preserve supported view geometry before any producer can select a data-copy kernel.
// Consumer adaptation and storage-isolation decisions belong to EdgeLayoutPlanner.
FX_REGISTER_OP_LOWERING(90, std::make_unique<fx_converter::ViewLayoutOpLowering>());
