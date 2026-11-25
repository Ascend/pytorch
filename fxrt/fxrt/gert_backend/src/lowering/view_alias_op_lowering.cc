/**
 * ViewAlias OpLowering — GE-aligned metadata reshape
 * (Reshape / Squeeze / Unsqueeze / Flatten / alias).
 *
 * Reuses the input device address and rebuilds a Tensor* with a new StorageShape
 * (not IdentityAlias, which keeps the same shape). Downstream Path C SplitDataTensor
 * reads runtime Tensor StorageShape, so a shape-only TensorRef update is insufficient.
 * Transpose / permute as_strided views keep the parent physical StorageShape
 * (kPermute); the selected consumer's edge planner adds a data conversion when required.
 *
 * RT chain (GE-aligned):
 *   (shape_producer, addr) dual-value when the input already unpacked
 *     -> InferShape / Const StorageShape
 *     -> consumer boxes via EnsureTensorRefBoxed (Path C tiling / DualStage / Autofuse)
 *   Graph Data: one cached SplitDataTensor, then same BuildRef.
 *
 * No Path B / no InstallFx*; launch/prepare/mem_to_free = null.
 * CNI is still emitted (opp_type=Reshape) so InferShape OutputsCreator can resolve
 * ComputeNodeInfo / output TensorDesc — skip_cni would leave infer_op without CNI index.
 */

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lowering/op_lowering.h"
#include "shape/shape_list_utils.h"

#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/graph_builder_utils.h"
#include "exe_graph/graph_skeleton_builder.h"
#include "ir/value/value.h"
#include "lowering/common/view_utils.h"
#include "lowering/common/wiring.h"
#include "register/op_catalog.h"
#include "lowering/op_lowering_registry.h"
#include "mrt_ge/mrt_node_utils.h"
#include "ops/op_def/ops_name.h"
#include "shape/fx_expr_infer_shape.h"
#include "shape/mrt_to_ge_expr.h"

namespace fx_converter {

// After Path C (100), before IdentityAlias (200) / DualStage (1000).
constexpr int kViewAliasPriority = 150;

namespace {

gert::OpImplRegisterV2::InferShapeKernelFunc g_view_infer_func = nullptr;

bool IsViewAliasOp(fxrt::ops::Op op) {
  return op == fxrt::ops::Op_view || op == fxrt::ops::Op_reshape || op == fxrt::ops::Op_unsqueeze_view ||
         op == fxrt::ops::Op_unsqueeze || op == fxrt::ops::Op_squeeze_view || op == fxrt::ops::Op_squeeze ||
         op == fxrt::ops::Op_squeeze_dim || op == fxrt::ops::Op_flatten_view || op == fxrt::ops::Op_alias ||
         op == fxrt::ops::Op_as_strided_view;
}

// CNI opp_type for InferShape OutputsCreator (ComputeNodeInfo). Flatten* / Reshape share gelocal.
const char *ViewAliasCniOppType(fxrt::ops::Op op) {
  if (op == fxrt::ops::Op_flatten_view) {
    return "Flatten";
  }
  if (op == fxrt::ops::Op_alias) {
    return "Identity";
  }
  return "Reshape";
}

// view/reshape size list from ABI0 int_list (Tuple<int>). All dims must be >= 0
// (no -1 infer); used as Const StorageShape when symbolic InferShape would fail
// under dynamic=True free-var seeds with broken shared_env Expressions.
static bool TryCaptureStaticViewSizeList(const fxrt::ir::NodePtr &node, std::vector<int64_t> *size) {
  if (node == nullptr || size == nullptr) {
    return false;
  }
  if (node->op != fxrt::ops::Op_view && node->op != fxrt::ops::Op_reshape) {
    return false;  // Squeeze axes and flatten dimensions are not an output-size tuple.
  }
  for (size_t i = 1; i < node->inputs.size(); ++i) {
    const auto &in = node->inputs[i];
    if (in == nullptr || in->output == nullptr || !in->output->IsTuple()) {
      continue;
    }
    std::vector<int64_t> dims = ReadIntListSymbolSafe(in->output->ToTuple());
    if (dims.empty()) {
      continue;
    }
    if (std::any_of(dims.begin(), dims.end(), [](int64_t d) { return d < 0; })) {
      return false;
    }
    *size = std::move(dims);
    return true;
  }
  return false;
}

// Export packs as_strided as int_list=[rank]+size+stride+[offset] (rebuilds as Tuple).
static bool ReadStaticAsStridedLayout(const fxrt::ir::NodePtr &node, std::vector<int64_t> *size,
                                      std::vector<int64_t> *stride, int64_t *offset, std::string *err) {
  if (size == nullptr || stride == nullptr || offset == nullptr) {
    return false;
  }
  if (TryReadAsStridedLayout(node, size, stride, offset)) {
    return true;
  }
  if (err != nullptr) {
    *err = "view-alias as_strided: layout must have static size/stride/offset metadata";
  }
  return false;
}

enum class AsStridedViewKind : uint8_t { kReject = 0, kContiguous = 1, kBroadcast = 2, kPermute = 3 };

// True when every static, non-unit dim of the view strides exactly like a row-major
// tensor of that size (wildcard dims are skipped). False reports non-contiguous strides.
static bool StridesAgreeWithContiguous(const std::vector<int64_t> &size, const std::vector<int64_t> &stride,
                                       const std::vector<int64_t> &expect_stride, std::string *err) {
  for (size_t i = 0; i < size.size(); ++i) {
    if (size[i] == 0) {
      break;
    }
    if (size[i] == 1) {
      continue;
    }
    if (size[i] < 0 || stride[i] < 0 || expect_stride[i] < 0) {
      continue;
    }
    if (stride[i] != expect_stride[i]) {
      if (err != nullptr) {
        *err = "view-alias as_strided: non-contiguous strides; Path C AsStrided not wired (refuse silent ViewAlias)";
      }
      return false;
    }
  }
  return true;
}

// Contiguous degrade, broadcast (stride 0), or permute of contiguous parent. Else Path C.
static AsStridedViewKind ClassifyAsStridedView(const fxrt::ir::NodePtr &node, const TensorRef &in,
                                               std::vector<int64_t> *size, std::vector<int64_t> *stride,
                                               std::string *err) {
  int64_t offset = 0;
  if (!ReadStaticAsStridedLayout(node, size, stride, &offset, err)) {
    return AsStridedViewKind::kReject;
  }
  const LayoutKind kind = ClassifyAsStridedLayout(PhysicalShape(in), *size, *stride, offset);
  if (kind == LayoutKind::kBroadcast) {
    return AsStridedViewKind::kBroadcast;
  }
  if (kind == LayoutKind::kPermute) {
    return AsStridedViewKind::kPermute;
  }
  if (offset != 0) {
    if (err != nullptr) {
      *err = "view-alias as_strided: storage_offset!=0 unsupported (got " + std::to_string(offset) + ")";
    }
    return AsStridedViewKind::kReject;
  }
  std::vector<int64_t> expect_stride;
  if (!ContiguousRowMajorStrides(*size, &expect_stride)) {
    if (err != nullptr) {
      *err = "view-alias as_strided: cannot compute contiguous strides for size";
    }
    return AsStridedViewKind::kReject;
  }
  if (!StridesAgreeWithContiguous(*size, *stride, expect_stride, err)) {
    return AsStridedViewKind::kReject;
  }
  const int64_t out_n = NumelOfShape(*size);
  const int64_t in_n = NumelOfShape(in.shape);
  if (out_n < 0 || in_n < 0) {
    return AsStridedViewKind::kContiguous;
  }
  if (in_n >= 0 && out_n != in_n) {
    if (err != nullptr) {
      *err = "view-alias as_strided: numel mismatch in=" + std::to_string(in_n) + " out=" + std::to_string(out_n);
    }
    return AsStridedViewKind::kReject;
  }
  return AsStridedViewKind::kContiguous;
}

// The kernel walks the base buffer itself, so all this has to hold is that the base address
// is where it starts from: as_strided composes storage offsets, and only the base address is
// handed over.
static bool ValidateAsStridedKernelAlias(const fxrt::ir::NodePtr &node, std::string *err) {
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  int64_t offset = 0;
  if (!TryReadAsStridedLayout(node, &size, &stride, &offset)) {
    if (err != nullptr) {
      *err = "view-alias as_strided: unreadable layout on a compiled-kernel argument";
    }
    return false;
  }
  if (offset != 0) {
    if (err != nullptr) {
      *err = "view-alias as_strided: compiled-kernel argument at storage_offset " + std::to_string(offset) +
             "; only the base address reaches the kernel, so it would start at the wrong element";
    }
    return false;
  }
  return true;
}

static bool IsCompiledKernelOnly(const fxrt::ir::NodePtr &node, const LoweringContext &ctx) {
  return ctx.compiled_kernel_only != nullptr && ctx.compiled_kernel_only->count(node.get()) != 0U;
}

// Prefer Const StorageShape when size is statically known (as_strided packed size,
// or view/reshape ABI0 int_list). Avoids FxExprInferFromRule GetResult failures when
// Dynamo dynamic=True leaves free-var seeds but reshape sizes are concrete ints.
bool ResolveViewAliasShapeProducer(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx,
                                   const TensorRef &in, const ge::TensorDesc &td_o, const ge::TensorDesc &empty_td,
                                   fx_converter::Node **shape_producer, uint64_t *s_shape, std::vector<int64_t> *const_size,
                                   bool *use_const_shape, std::vector<fx_converter::Node *> *cni_nodes, std::string *err) {
  ge::Graph *main = ctx.main;
  uint64_t &sym_gen = *ctx.sym_gen;
  *use_const_shape = false;
  if (node->op == fxrt::ops::Op_as_strided_view) {
    std::vector<int64_t> packed_stride;
    int64_t packed_off = 0;
    if (!ReadStaticAsStridedLayout(node, const_size, &packed_stride, &packed_off, err)) {
      return false;
    }
    const LayoutKind packed_kind = ClassifyAsStridedLayout(PhysicalShape(in), *const_size, packed_stride, packed_off);
    if (packed_kind == LayoutKind::kBroadcast || packed_kind == LayoutKind::kPermute) {
      *const_size = in.storage_shape.empty() ? in.shape : in.storage_shape;
      if (in.HasUnpackedPair()) {
        *shape_producer = in.unpacked.shape.node;
        *s_shape = in.unpacked.shape.sym;
        *use_const_shape = true;
        return true;
      }
      if (std::any_of(const_size->begin(), const_size->end(), [](int64_t d) { return d < 0; })) {
        if (err != nullptr) {
          *err = "view-alias broadcast: parent storage is not unpacked and not static";
        }
        return false;
      }
      *use_const_shape = true;
      return AddConcreteStorageShapeConst(main, ctx.built, tag, *const_size, sym_gen, ctx.kernel_nodes,
                                          shape_producer, s_shape, err);
    }
    const bool static_size =
        !std::any_of(const_size->begin(), const_size->end(), [](int64_t d) { return d < 0; });
    *use_const_shape = static_size;
    if (!static_size) {
      const_size->clear();
    }
  } else if (TryCaptureStaticViewSizeList(node, const_size)) {
    *use_const_shape = true;
  } else {
    // Static squeeze/unsqueeze/flatten also have exact output geometry. A scalar
    // is a valid empty shape, and does not require a free-variable shape rule.
    *const_size = ShapeOfNode(node);
    *use_const_shape = std::all_of(const_size->begin(), const_size->end(), [](int64_t d) { return d >= 0; });
  }
  if (*use_const_shape) {
    return AddConcreteStorageShapeConst(main, ctx.built, tag, *const_size, sym_gen, ctx.kernel_nodes, shape_producer,
                                        s_shape, err);
  }
  auto rule = BuildOutputSymbolicShapeRule(node, ctx.symbol_env, fxrt::ops::ToStr(node->op));
  if (rule == nullptr) {
    if (err != nullptr) {
      *err = "view-alias lowering: no symbolic output shape / SymbolEnv seeds";
    }
    return false;
  }
  if (g_view_infer_func == nullptr) {
    g_view_infer_func = GetFxExprInferShapeFunc();
  }
  const FxShapeRule *rule_ptr = rule.get();
  ctx.built->shape_rules.push_back(std::move(rule));
  fx_converter::Node *c_rule =
      AddConstNode(main, "const_shape_rule_" + tag, &rule_ptr, sizeof(rule_ptr), false, sym_gen,
                   FxrtConstType::kVoidPtr);
  ctx.kernel_nodes->push_back(c_rule);
  fx_converter::Node *c_infer =
      AddConstNode(main, "const_infer_func_" + tag, &g_view_infer_func, sizeof(g_view_infer_func), false, sym_gen,
                   FxrtConstType::kKernelFuncPtr);
  ctx.kernel_nodes->push_back(c_infer);
  *shape_producer =
      AddInferShapeNode(main, tag, std::vector<TensorRef>{in}, td_o, empty_td, c_rule, c_infer, sym_gen, s_shape);
  if (ctx.bind_symbols != nullptr) {
    (void)AddControlEdge(main, ctx.bind_symbols, *shape_producer);
  }
  ctx.kernel_nodes->push_back(*shape_producer);
  cni_nodes->push_back(*shape_producer);
  return true;
}

bool WireViewAliasSplitAndBuildRef(const std::string &tag, LoweringContext &ctx, const TensorRef &in,
                                   fx_converter::Node *shape_producer, uint64_t s_shape, const ge::TensorDesc &td_o,
                                   const ge::TensorDesc &empty_td, ge::DataType out_dtype, fx_converter::Node **split,
                                   fx_converter::Node **build_ref, uint64_t *s_out_tensor, TensorEndpoint *addr_ep) {
  (void)tag;
  (void)shape_producer;
  (void)s_shape;
  (void)td_o;
  (void)empty_td;
  (void)out_dtype;
  if (split == nullptr || build_ref == nullptr || s_out_tensor == nullptr || addr_ep == nullptr) {
    return false;
  }
  if (in.HasUnpackedPair()) {
    *addr_ep = in.unpacked.addr;
    *split = nullptr;
  } else {
    *split = GetOrCreateSplitDataTensor(ctx, in, "view_" + tag, 0, nullptr, &addr_ep->sym);
    addr_ep->node = *split;
    addr_ep->out = 1;
  }
  if (!addr_ep->Valid()) {
    return false;
  }
  // A view is metadata-only: propagate (new shape, root address). Materialize a
  // boxed Tensor only at a consumer ABI boundary via EnsureTensorRefBoxed.
  *build_ref = nullptr;
  *s_out_tensor = 0;
  return true;
}

bool AttachLogicalShapeProducer(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx,
                                const TensorRef &in, const ge::TensorDesc &td_o, const ge::TensorDesc &empty_td,
                                const std::vector<int64_t> &packed_size, LoweringResult *out, std::string *err) {
  if (out == nullptr) {
    return false;
  }
  out->outputs.resize(1);  // Idempotent: FillViewAliasResult may already have sized it.
  TensorRef &out0 = out->outputs[0];
  const bool static_size = std::none_of(packed_size.begin(), packed_size.end(), [](int64_t d) { return d < 0; });
  if (static_size) {
    return AddConcreteStorageShapeConst(ctx.main, ctx.built, tag + "_logical", packed_size, *ctx.sym_gen,
                                        ctx.kernel_nodes, &out0.logical_shape_ep.node, &out0.logical_shape_ep.sym, err);
  }
  auto rule = BuildOutputSymbolicShapeRule(node, ctx.symbol_env, fxrt::ops::ToStr(node->op));
  if (rule == nullptr) {
    if (err != nullptr) {
      *err = "view-alias logical: no symbolic output shape / SymbolEnv seeds";
    }
    return false;
  }
  if (g_view_infer_func == nullptr) {
    g_view_infer_func = GetFxExprInferShapeFunc();
  }
  const FxShapeRule *rule_ptr = rule.get();
  ctx.built->shape_rules.push_back(std::move(rule));
  fx_converter::Node *c_rule =
      AddConstNode(ctx.main, "const_logical_shape_rule_" + tag, &rule_ptr, sizeof(rule_ptr), false, *ctx.sym_gen,
                   FxrtConstType::kVoidPtr);
  ctx.kernel_nodes->push_back(c_rule);
  fx_converter::Node *c_infer = AddConstNode(ctx.main, "const_logical_infer_func_" + tag, &g_view_infer_func,
                                      sizeof(g_view_infer_func), false, *ctx.sym_gen,
                                      FxrtConstType::kKernelFuncPtr);
  ctx.kernel_nodes->push_back(c_infer);
  uint64_t s_logical = 0;
  fx_converter::Node *logical_infer =
      AddInferShapeNode(ctx.main, tag + "_logical", std::vector<TensorRef>{in}, td_o, empty_td, c_rule, c_infer,
                        *ctx.sym_gen, &s_logical);
  out0.logical_shape_ep = TensorEndpoint{logical_infer, 0, s_logical};
  if (ctx.bind_symbols != nullptr && out0.logical_shape_ep.node != nullptr) {
    (void)AddControlEdge(ctx.main, ctx.bind_symbols, out0.logical_shape_ep.node);
  }
  if (out0.logical_shape_ep.node != nullptr) {
    ctx.kernel_nodes->push_back(out0.logical_shape_ep.node);
  }
  return out0.logical_shape_ep.node != nullptr;
}

void FillViewAliasResult(const fxrt::ir::NodePtr &node, const TensorRef &in, fx_converter::Node *split,
                         fx_converter::Node *build_ref, uint64_t s_out_tensor, bool use_const_shape,
                         const std::vector<int64_t> &const_size, const std::vector<int64_t> &out_shape,
                         ge::DataType out_dtype, std::vector<fx_converter::Node *> cni_nodes, fx_converter::Node *shape_producer,
                         uint64_t s_shape, const TensorEndpoint &addr_ep, LoweringResult *out) {
  out->outputs.resize(1);
  TensorRef &out0 = out->outputs[0];
  out0.producer = build_ref;
  out0.symbol = s_out_tensor;
  out0.shape = use_const_shape ? const_size : out_shape;
  out0.dtype = out_dtype;
  out0.format = in.format;
  out0.unpacked.shape = TensorEndpoint{shape_producer, 0, s_shape};
  out0.unpacked.addr = addr_ep;
  out->launch = nullptr;
  out->prepare = nullptr;
  // Minimal CNI for InferShape OutputsCreator (GetOutputDesc via ComputeNodeInfo).
  out->skip_cni = false;
  out->cni.op_type = ViewAliasCniOppType(node->op);
  out->cni.cni_io = CniIoStyle::kIndexedFrom0;
  out->cni.behavior = nullptr;
  out->cni.input_shapes = {in.shape};
  out->cni.input_dtypes = {static_cast<int32_t>(in.dtype)};
  out->cni.input_formats = {static_cast<int32_t>(in.format)};
  out->cni.output_shape = use_const_shape ? const_size : out_shape;
  out->cni.output_dtype = static_cast<int32_t>(out_dtype);
  out->cni.output_formats = {static_cast<int32_t>(in.format)};
  if (split != nullptr) {
    cni_nodes.push_back(split);
  }
  if (build_ref != nullptr) {
    cni_nodes.push_back(build_ref);
  }
  out->cni.compute_nodes = std::move(cni_nodes);
}

}  // namespace

class ViewAliasOpLowering : public OpLowering {
 public:
  const char *Name() const override { return "ViewAlias"; }

  InputLayoutKind InputLayout(const fxrt::ir::NodePtr &, size_t) const override {
    return InputLayoutKind::kMetadata;
  }

  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *ctx) const override {
    if (node == nullptr || !IsViewAliasOp(node->op)) {
      return false;
    }
    if (node->op == fxrt::ops::Op_as_strided_view) {
      for (size_t i = 1; i < node->inputs.size(); ++i) {
        const auto &input = node->inputs[i];
        if (input != nullptr && input->output != nullptr && input->output->IsTuple() &&
            IntListHasSymbol(input->output->ToTuple())) {
          // Symbolic geometry is preserved by ViewLayout / runtime TensorV2 boxing;
          // ViewAlias only accepts compile-time-proven geometry.
          return false;
        }
        if (input != nullptr && input->output != nullptr && input->output->IsSymbol()) return false;
      }
      std::vector<int64_t> size;
      std::vector<int64_t> stride;
      int64_t offset = 0;
      if (TryReadAsStridedLayout(node, &size, &stride, &offset) && offset == -1) return false;
    }
    // Dynamic stride/offset geometry is a Path C input contract. Let the AICore
    // AsStrided lowering consume the symbolic payload instead of claiming the
    // node here and rejecting it after ViewAlias has already been selected.
    const bool path_c = node->op == fxrt::ops::Op_as_strided_view && IsAsStridedPathC(node);
    return node->op != fxrt::ops::Op_as_strided_view || ctx == nullptr || !path_c;
  }

  // view and reshape keep distinct IR semantics. make_tuple and tuple_getitem are
  // graph-structure markers the execute graph consumes; they have no catalog row.
  std::vector<std::string> SupportedLogicalOps() const override {
    return {"view",           "reshape",      "unsqueeze_view", "unsqueeze",    "squeeze_view", "squeeze",
            "squeeze_dim",    "flatten_view", "alias",          "make_tuple",   "tuple_getitem"};
  }

  bool Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx, LoweringResult *out,
             std::string *err) override {
    if (out == nullptr || !ValidateLoweringContext(ctx, false, "view-alias lowering", err)) {
      return false;
    }
    TensorRef in;
    std::vector<int64_t> packed_size;
    std::vector<int64_t> packed_stride;
    AsStridedViewKind as_kind = AsStridedViewKind::kContiguous;
    if (node->op == fxrt::ops::Op_as_strided_view) {
      // as_strided is storage-addressed: alias the buffer its size / stride / offset were
      // computed against, which is the storage root of the input rather than a view of it
      // that the converter may have materialized into a buffer of its own.
      if (!ResolveAsStridedBaseTensor(node, ctx, &in, nullptr, err)) {
        return false;
      }
      // A compiled kernel does its own index arithmetic, so an as_strided nothing else reads
      // is an alias by definition -- the strides describe how the kernel will walk the base,
      // not a layout anyone has to reproduce. Classify may still mark broadcast / permute
      // so DualStage sees physical storage; other non-contiguous layouts stay a plain alias.
      if (IsCompiledKernelOnly(node, ctx)) {
        if (!ValidateAsStridedKernelAlias(node, err)) {
          return false;
        }
        std::string classify_err;
        as_kind = ClassifyAsStridedView(node, in, &packed_size, &packed_stride, &classify_err);
        if (as_kind == AsStridedViewKind::kReject) {
          as_kind = AsStridedViewKind::kContiguous;
        }
      } else {
        as_kind = ClassifyAsStridedView(node, in, &packed_size, &packed_stride, err);
        if (as_kind == AsStridedViewKind::kReject) {
          return false;
        }
      }
    } else if (!ResolveInputTensor(node, ctx, &in, err)) {
      return false;
    }

    const ge::TensorDesc empty_td;
    const std::vector<int64_t> out_shape = ShapeOfNode(node);
    // Views do not change element type. IR dtype can disagree with storage
    // (TopKV2 indices are INT32 while aten.topk getitem is annotated i64);
    // follow the buffer we actually alias.
    const ge::DataType out_dtype = in.dtype;
    const ge::TensorDesc td_o = MakeTensorDesc(out_shape, out_dtype, in.format);

    fx_converter::Node *shape_producer = nullptr;
    uint64_t s_shape = 0;
    std::vector<fx_converter::Node *> cni_nodes;
    std::vector<int64_t> const_size;
    bool use_const_shape = false;
    if (!ResolveViewAliasShapeProducer(node, tag, ctx, in, td_o, empty_td, &shape_producer, &s_shape, &const_size,
                                       &use_const_shape, &cni_nodes, err)) {
      return false;
    }

    fx_converter::Node *split = nullptr;
    fx_converter::Node *build_ref = nullptr;
    uint64_t s_out_tensor = 0;
    TensorEndpoint addr_ep;
    if (!WireViewAliasSplitAndBuildRef(tag, ctx, in, shape_producer, s_shape, td_o, empty_td, out_dtype, &split,
                                       &build_ref, &s_out_tensor, &addr_ep)) {
      return false;
    }

    FillViewAliasResult(node, in, split, build_ref, s_out_tensor, use_const_shape, const_size, out_shape, out_dtype,
                        std::move(cni_nodes), shape_producer, s_shape, addr_ep, out);
    TensorRef &out0 = out->outputs[0];  // Sized by FillViewAliasResult above.
    // A contiguous view reinterprets the producer buffer. Fallback outputs are
    // adopted Torch storages, not L2 blocks; allocating a fresh model output and
    // copying would fault. Keep the adopted address and publish the view shape.
    if (as_kind != AsStridedViewKind::kBroadcast && as_kind != AsStridedViewKind::kPermute) {
      out0.can_adopt_for_model_output = in.can_adopt_for_model_output;
    }
    out0.allocation = in.allocation;
    out0.feed_index = in.feed_index;
    if (as_kind == AsStridedViewKind::kBroadcast || as_kind == AsStridedViewKind::kPermute) {
      out0.storage_shape = in.storage_shape.empty() ? in.shape : in.storage_shape;
      out0.logical_shape = packed_size;
      out0.stride = packed_stride;
      out0.storage_offset = in.storage_offset;
      if (!AttachLogicalShapeProducer(node, tag, ctx, in, td_o, empty_td, packed_size, out, err)) {
        return false;
      }
    } else if (!IsDenseTensorRef(in)) {
      const auto &logical = use_const_shape ? const_size : out_shape;
      if (!ComputeReshapeViewStrides(LogicalShape(in), in.stride, logical, &out0.stride)) {
        if (err != nullptr) *err = "view-alias: input layout cannot represent the target shape";
        return false;
      }
      out0.logical_shape = logical;
      out0.storage_shape = PhysicalShape(in);
      out0.shape = PhysicalShape(in);
      out0.storage_offset = in.storage_offset;
      // Preserve physical StorageShape for downstream transpose-aware ABI; logical
      // shape remains a separate edge for InferShape, just as for as_strided.
      out0.unpacked.shape = in.unpacked.shape;
      if (!AttachLogicalShapeProducer(node, tag, ctx, in, td_o, empty_td, logical, out, err)) return false;
    } else if (out0.stride.empty()) {
      (void)ContiguousRowMajorStrides(out0.shape, &out0.stride);
      out0.proven_dense = true;
      out0.storage_shape = out0.shape;
      out0.storage_offset = in.storage_offset;
    }
    // Ordinary view operations retain the source allocation's per-Execute
    // origin. as_strided carries an explicit storage-coordinate offset instead.
    if (node->op != fxrt::ops::Op_as_strided_view) {
      out0.runtime_storage_offset = in.runtime_storage_offset;
    }
    return true;
  }
};

// Moving only literal size-one axes preserves the element iteration order of a
// dense input. Unknown dimensions retain their order, so this proof also holds
// for dynamic batches. Reuse ViewAlias's runtime shape rule without a Transpose
// task or an absolute FakeTensor stride substitution.
class SingletonPermuteOpLowering final : public ViewAliasOpLowering {
 public:
  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *ctx) const override {
    if (ctx == nullptr || node == nullptr || node->op != fxrt::ops::Op_permute_view ||
        node->attrs.count("fxrt_write_dense") != 0U || node->inputs.size() != 2U ||
        node->inputs[1] == nullptr || node->inputs[1]->output == nullptr ||
        !node->inputs[1]->output->IsTuple()) return false;
    TensorRef input;
    if (!ResolveInputTensor(node, *ctx, &input, nullptr) || !IsDenseTensorRef(input)) return false;
    const auto &shape = LogicalShape(input);
    const auto permutation = ReadIntListSymbolSafe(node->inputs[1]->output->ToTuple());
    if (permutation.size() != shape.size()) return false;
    std::vector<bool> used(shape.size(), false);
    size_t next = 0U;
    for (int64_t axis : permutation) {
      if (axis < 0) axis += static_cast<int64_t>(shape.size());
      if (axis < 0 || static_cast<size_t>(axis) >= shape.size() || used[axis]) return false;
      used[axis] = true;
      if (shape[axis] == 1) continue;
      while (next < shape.size() && shape[next] == 1) ++next;
      if (static_cast<size_t>(axis) != next++) return false;
    }
    return true;
  }

  std::vector<std::string> SupportedLogicalOps() const override { return {}; }
};

}  // namespace fx_converter

// Before ViewLayout (90) and the Path C data-copy lowerer (100).
FX_REGISTER_OP_LOWERING(80, std::make_unique<::fx_converter::SingletonPermuteOpLowering>());
FX_REGISTER_OP_LOWERING(::fx_converter::kViewAliasPriority,
                        std::make_unique<::fx_converter::ViewAliasOpLowering>());
