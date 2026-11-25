/**
 * AICore binary-reuse OpLowering (path C) — entry orchestration.
 *
 * Convert-time: match the catalog row, collect/materialize inputs, select the CANN
 *               prebuilt .o (simplifiedKey), then wire the GE-isomorphic RT sequence.
 * Execute-time: FxAiCoreTiling calls the CANN OpImpl tiling function, then
 * FxAiCoreLaunch submits the kernel with aclrtLaunchKernelWithHostArgs.
 *
 * Arity 0 is variadic packed-x. Scalar IR inputs for eq_scalar / ge_scalar / ne
 * are materialized as 1-element device tensors.
 *
 * File split: input collection -> aicore_input_collect; scalar / shape-list
 * materialization -> aicore_input_materialize; RT wiring -> aicore_rt_wiring.
 */

#include "lowering/aicore/aicore_binary_op_lowering.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/graph_builder_utils.h"
#include "graph/utils/type_utils.h"
#include "ir/value/value.h"
#include "lowering/aicore/aicore_bin_artifacts.h"
#include "lowering/aicore/aicore_binary_kernel_selector.h"
#include "lowering/aicore/aicore_input_collect.h"
#include "lowering/aicore/aicore_input_materialize.h"
#include "lowering/aicore/aicore_op_work.h"
#include "lowering/aicore/aicore_rt_wiring.h"
#include "lowering/catalog_attrs.h"
#include "lowering/common/wiring.h"
#include "lowering/op_lowering_registry.h"
#include "mrt_ge/mrt_node_utils.h"
#include "opp/opp_registry.h"
#include "ops/op_def/ops_name.h"
#include "register/op_catalog.h"
#include "shape/fx_expr_infer_shape.h"
#include "shape/mrt_to_ge_expr.h"

namespace fx_converter {
namespace {

// InferShape kernel func slot: filled lazily from the OpImpl registry, shared by every
// per-op Const snapshot (Const copies the slot value; the slot itself stays stable).
gert::OpImplRegisterV2::InferShapeKernelFunc &AiCoreInferFuncSlot() {
  static gert::OpImplRegisterV2::InferShapeKernelFunc func = nullptr;
  if (func == nullptr) {
    func = GetFxExprInferShapeFunc();
  }
  return func;
}

bool RefIndexInRange(int32_t index, size_t limit) {
  return index >= 0 && static_cast<size_t>(index) < limit;
}

bool ValidateOneOutputRef(const OutputRef &ref, size_t nout, size_t nin, std::vector<bool> *seen_out,
                          std::vector<bool> *seen_in, const std::string &tag, std::string *err) {
  if (!RefIndexInRange(ref.output, nout)) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: ref output " + std::to_string(ref.output) +
             " is outside the outputs of '" + tag + "'";
    }
    return false;
  }
  if (!RefIndexInRange(ref.input, nin)) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: ref input " + std::to_string(ref.input) +
             " is outside the collected inputs of '" + tag + "'";
    }
    return false;
  }
  if ((*seen_out)[static_cast<size_t>(ref.output)] || (*seen_in)[static_cast<size_t>(ref.input)]) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: duplicate ref output " + std::to_string(ref.output) + " or input " +
             std::to_string(ref.input) + " on '" + tag + "'";
    }
    return false;
  }
  (*seen_out)[static_cast<size_t>(ref.output)] = true;
  (*seen_in)[static_cast<size_t>(ref.input)] = true;
  return true;
}

bool ValidateOutputRefs(const AiCoreOpWork &work, std::string *err) {
  const OpCatalogEntry &entry = work.entry;
  if (entry.output_ref_count == 0) {
    return true;
  }
  if (entry.output_refs == nullptr) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: '" + work.tag + "' declares output refs without storage";
    }
    return false;
  }
  std::vector<bool> seen_out(work.dtypes.size(), false);
  std::vector<bool> seen_in(work.ins.size(), false);
  for (uint8_t i = 0; i < entry.output_ref_count; ++i) {
    if (!ValidateOneOutputRef(entry.output_refs[i], work.dtypes.size(), work.ins.size(), &seen_out, &seen_in,
                              work.tag, err)) {
      return false;
    }
  }
  return true;
}

AiCoreOppBinding ResolveAiCoreOppBinding(const AiCoreOpWork &work) {
  const OpCatalogEntry &entry = work.entry;
  AiCoreOppBinding b;
  b.opp_type = entry.opp_type;
  b.opp_input_names = entry.opp_input_names;
  b.pack_entry = &entry;
  if (work.extras.recipe.use_as_strided) {
    // TBE kernel always has storage_offset GM; the 910B simplifiedKey omits that
    // optional slot, so select with the 3 required inputs only.
    b.opp_type = kAsStridedOppType;
    b.opp_input_names = kAsStridedInputNames;
    b.select_input_count = kAsStridedSelectInputCount;
    b.pack_entry = nullptr;
  }
  return b;
}

// opp_output_dtypes CSV: a non-empty cell forces that output's GE dtype (TopKV2
// declares ",DT_INT32" because the indices bins are INT32).
bool ApplyCatalogOutputDtypes(AiCoreOpWork &work, std::string *err) {
  if (work.entry.opp_output_dtypes == nullptr) {
    return true;
  }
  const size_t declared = CatalogCsvCount(work.entry.opp_output_dtypes);
  for (size_t o = 0; o < work.dtypes.size() && o < declared; ++o) {
    const std::string cell = CatalogCsvField(work.entry.opp_output_dtypes, o);
    if (cell.empty()) {
      continue;
    }
    const ge::DataType dtype = ge::TypeUtils::SerialStringToDataType(cell);
    if (dtype == ge::DT_UNDEFINED) {
      if (err != nullptr) {
        *err = std::string("aicore-binary lowering: unknown opp_output_dtypes cell '") + cell + "' for '" +
               work.entry.logical_name + "'";
      }
      return false;
    }
    work.dtypes[o] = dtype;
  }
  return true;
}

// ND plus the contiguous layouts torch reports for a dense buffer. Anything else
// (FRACTAL_NZ and the rest) is a private storage format.
bool IsPrivateStorageFormat(ge::Format format) {
  return format != ge::FORMAT_ND && format != ge::FORMAT_NCHW && format != ge::FORMAT_NHWC &&
         format != ge::FORMAT_NCDHW;
}

// No AiCore binary here pairs a private input format with an ND new output.
// Stop before kernel select. aclnn reads the same feed format on its own path.
bool RejectPrivateFormat(const AiCoreOpWork &work, LoweringContext &ctx, std::string *err) {
  for (size_t i = 0; i < work.ins.size(); ++i) {
    const int32_t feed = work.ins[i].feed_index;
    if (feed >= 0 && ctx.built != nullptr &&
        static_cast<size_t>(feed) < ctx.built->input_requires_base_format.size()) {
      ctx.built->input_requires_base_format[static_cast<size_t>(feed)] = true;
    }
    if (!IsPrivateStorageFormat(work.ins[i].format)) {
      continue;
    }
    if (err != nullptr) {
      *err = std::string("aicore-binary lowering: special format is not supported; '") +
             work.entry.logical_name + "' input " + std::to_string(i) + " carries format " +
             std::to_string(static_cast<int32_t>(work.ins[i].format));
    }
    return false;
  }
  return true;
}

bool ResolveAiCoreOutputs(AiCoreOpWork &work, std::string *err) {
  const size_t nout = CatalogOutputArity(work.entry);
  const std::vector<fxrt::ir::TensorPtr> slots = TensorsOfNode(work.node);
  work.out_shapes.assign(nout, {-1, -1});
  work.dtypes.assign(nout, ge::DT_FLOAT);
  const size_t n = std::min(nout, slots.size());
  for (size_t o = 0; o < n; ++o) {
    work.out_shapes[o] = ShapeOfTensor(slots[o]);
    work.dtypes[o] = DtypeOfTensor(slots[o]);
  }
  if (!ApplyCatalogOutputDtypes(work, err)) {
    return false;
  }
  // Expand (broadcast_target + shape-list) keeps the data-buffer dtype. aten.topk
  // indices are INT32 in Path C but the expand IR may still say i64; OPP Expand
  // bins require x/y the same dtype (3,2/3,2), not 3,2/9,2.
  if (work.entry.shape_input_index >= 0 && work.entry.shape_list_broadcast_target && !work.ins.empty()) {
    std::fill(work.dtypes.begin(), work.dtypes.end(), work.ins[0].dtype);
  }
  for (uint8_t i = 0; i < work.entry.attr_count; ++i) {
    const GeAttr &attr = work.entry.attrs[i];
    if (attr.derived == GeAttrDerived::kOutputDtype && attr.name != nullptr) {
      work.extras.SetInt(attr.name, static_cast<int64_t>(work.dtypes[0]));
    }
  }
  return true;
}

bool HasBoolExtra(const CatalogAttrExtras &extras, const char *name) {
  if (name == nullptr) {
    return false;
  }
  for (uint8_t idx = 0; idx < extras.count; ++idx) {
    const CatalogNamedAttr &attr = extras.slots[idx];
    if (attr.name != nullptr && std::string(attr.name) == name &&
        (attr.kind == CatalogAttrKind::kBool || attr.kind == CatalogAttrKind::kInt)) {
      return true;
    }
  }
  return false;
}

bool LookupCatalogAttrBool(const OpCatalogEntry &entry, const char *name, bool *out) {
  if (entry.attrs == nullptr || name == nullptr || out == nullptr) {
    return false;
  }
  for (const GeAttr *attr = entry.attrs; attr->name != nullptr; ++attr) {
    if (std::string(attr->name) == name) {
      *out = attr->literal != 0;
      return true;
    }
  }
  return false;
}

// Captured extras win; otherwise the catalog attr literal. Neither means the
// "{name}" placeholder cannot be expanded — do not guess true.
bool ResolveSelectBool(const char *key, const AiCoreOpWork &work, bool &out, std::string *err) {
  if (HasBoolExtra(work.extras, key)) {
    out = work.extras.BoolOr(key, false);
    return true;
  }
  if (LookupCatalogAttrBool(work.entry, key, &out)) {
    return true;
  }
  if (err != nullptr) {
    *err = std::string("aicore-binary lowering: select suffix attr '") + key +
           "' has no captured value and no catalog default";
  }
  return false;
}

// Literal suffix, or "{bool_attr}" → "0"/"1" from extras (TopKV2: "{sorted},,{largest}").
bool ExpandSelectSuffix(const char *tmpl, const AiCoreOpWork &work, std::string &expanded, std::string *err) {
  expanded.clear();
  if (tmpl == nullptr || tmpl[0] == '\0') {
    return true;
  }
  const std::string text(tmpl);
  size_t i = 0;
  while (i < text.size()) {
    if (text[i] != '{') {
      expanded.push_back(text[i]);
      ++i;
      continue;
    }
    const size_t end = text.find('}', i);
    if (end == std::string::npos) {
      expanded.append(text.substr(i));
      break;
    }
    const std::string key = text.substr(i + 1, end - i - 1);
    bool value = false;
    if (!ResolveSelectBool(key.c_str(), work, value, err)) {
      return false;
    }
    expanded.push_back(value ? '1' : '0');
    i = end + 1;
  }
  return true;
}

bool FillBinarySelectAttrSuffix(const AiCoreOpWork &work, BinarySelectRequest &req, std::string *err) {
  const OpBehavior *behavior = work.entry.behavior;
  if (behavior != nullptr && behavior->select_suffix != nullptr) {
    return ExpandSelectSuffix(behavior->select_suffix, work, req.attr_key_suffix, err);
  }
  return true;
}

bool FillBinarySelectRequest(const AiCoreOpWork &work, BinarySelectRequest &req, std::string *err) {
  req.opp_type = work.bind.opp_type;
  if (work.entry.cni_io == CniIoStyle::kDynamicPackedX && !work.ins.empty()) {
    // GE dynamic packed-x keys encode one representative x slot carrying the packed
    // input count (Pack/d,p/dtype,format,N/...), not one slot per tensor.
    req.inputs.push_back(BinaryIoSpec{BinaryAbiDtype(work.entry, work.ins.front().dtype), ge::FORMAT_ND});
    req.packed_input_count = static_cast<int32_t>(work.ins.size());
  } else {
    req.inputs.reserve(work.ins.size());
    for (const auto &in : work.ins) {
      req.inputs.push_back(BinaryIoSpec{BinaryAbiDtype(work.entry, in.dtype), ge::FORMAT_ND});
    }
  }
  // Select key may carry fewer slots than launch io. Launch/CNI/tiling op description keep the full layout.
  //  - bind.select_input_count: 910B AsStrided keys are 3-in+1-out; storage_offset is
  //    optional in the key but still launched (4 launch inputs).
  //  - optional tail omitted from simplifiedKey (CrossEntropyLoss weight, AscendQuantV2
  //    offset). AsStrided rebinds opp_input_names, so recount those names.
  size_t select_in = req.inputs.size();
  if (work.bind.select_input_count != 0) {
    select_in = std::min(select_in, work.bind.select_input_count);
  }
  if (work.optional.has_optional) {
    const bool rebound = work.bind.opp_input_names != work.entry.opp_input_names;
    const size_t required = rebound ? CatalogCsvCount(work.bind.opp_input_names) : work.optional.required;
    select_in = std::min(select_in, required);
  }
  req.inputs.resize(select_in);
  const bool select_single = work.entry.behavior != nullptr && work.entry.behavior->select_single_output;
  const size_t nout = select_single ? 1U : work.dtypes.size();
  for (size_t i = 0; i < nout; ++i) {
    req.outputs.push_back(BinaryIoSpec{BinaryAbiDtype(work.entry, work.dtypes[i]), ge::FORMAT_ND});
  }
  req.prefer_high_performance = work.entry.behavior == nullptr || work.entry.behavior->prefer_high_performance;
  return FillBinarySelectAttrSuffix(work, req, err);
}

bool BindWorkOptional(AiCoreOpWork &work, std::string *err) {
  return BindAiCoreOptionalConnected(&work.optional, work.ins.size(), work.entry.logical_name, err);
}

bool SelectAndBuildAiCoreArtifacts(LoweringContext &ctx, AiCoreOpWork &work, std::string *err) {
  gert::OpImplSpaceRegistryV2Ptr reg;
  if (!EnsureOppRegistryReady(&reg) || reg == nullptr) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: OPP registry not ready";
    }
    return false;
  }
  BinarySelectRequest req;
  SelectedBinaryKernel selected;
  if (!FillBinarySelectRequest(work, req, err) || !SelectBinaryKernel(req, selected, err) ||
      !BuildAiCoreBinArtifacts(selected, work, err)) {
    return false;
  }
  ctx.built->binary_bindata_blobs.push_back(work.art.bindata_blob);
  ctx.built->binary_bindata_blobs.push_back(work.art.compute_node_desc);
  ctx.built->binary_bindata_blobs.push_back(work.art.args_info_desc);
  ctx.built->binary_kernels.push_back(std::move(selected));
  return true;
}

bool WireAiCoreShapeRuleConsts(AiCoreOpWork &work, LoweringContext &ctx, RtGraphBuilder &gb,
                               std::vector<RtGraphBuilder::Value> &c_rules, RtGraphBuilder::Value &c_infer,
                               std::string *err) {
  if (AiCoreInferFuncSlot() == nullptr) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: FxExprInferShape func is null";
    }
    return false;
  }
  // One rule per output slot: multi-output slots may carry distinct shapes (LayerNormV4
  // mean/rstd drop the trailing norm dims that y keeps). The InferShape kernel consumes
  // rule i for output slot i, so nothing is mirrored across slots.
  std::vector<fxrt::ir::TensorPtr> slot_tensors = TensorsOfNode(work.node);
  if (slot_tensors.empty()) {
    slot_tensors.push_back(nullptr);
  }
  c_rules.reserve(c_rules.size() + slot_tensors.size());
  for (size_t oi = 0; oi < slot_tensors.size(); ++oi) {
    std::shared_ptr<FxShapeRule> rule =
        BuildOutputSymbolicShapeRuleFromTensor(slot_tensors[oi], ctx.symbol_env, work.entry.logical_name);
    if (rule == nullptr && oi == 1U && std::string(work.entry.logical_name) == "topk" && !c_rules.empty()) {
      rule = BuildOutputSymbolicShapeRuleFromTensor(slot_tensors[0], ctx.symbol_env, work.entry.logical_name);
    }
    if (rule == nullptr) {
      if (err != nullptr) {
        *err = std::string("aicore-binary lowering: no symbolic shape rule for '") + work.entry.logical_name +
               "' output slot " + std::to_string(oi);
      }
      return false;
    }
    const FxShapeRule *rule_ptr = rule.get();
    ctx.built->shape_rules.push_back(std::move(rule));
    c_rules.push_back(
        gb.Const("const_shape_rule_" + work.tag + "_" + std::to_string(oi), &rule_ptr, sizeof(rule_ptr), false));
  }
  auto *infer_slot = &AiCoreInferFuncSlot();
  c_infer = gb.Const("const_infer_func_" + work.tag, infer_slot, sizeof(*infer_slot), false);
  return true;
}

std::vector<TensorRef> BuildAiCoreOutputTensorRefs(const AiCoreOpWork &work) {
  const size_t nout = work.dtypes.size();
  std::vector<TensorRef> outs(nout);
  for (size_t o = 0; o < nout; ++o) {
    outs[o] = work.stage.OutputRef(o, work.out_shapes[o], work.dtypes[o]);
    const int32_t input = CatalogRefInput(work.entry, o);
    if (!work.stage.outs[o].owns_addr && input >= 0 && static_cast<size_t>(input) < work.ins.size()) {
      const TensorRef &source = work.ins[static_cast<size_t>(input)];
      outs[o].allocation = source.allocation.Valid() ? source.allocation : source.unpacked.addr;
      outs[o].feed_index = source.feed_index;
    }
  }
  return outs;
}

void NormalizeCapturedAxis(AiCoreOpWork &work) {
  const OpBehavior *behavior = work.entry.behavior;
  if (behavior == nullptr || behavior->axis_rank_delta == OpBehavior::kNoAxisNormalize ||
      work.axis == std::numeric_limits<int64_t>::min() || work.axis >= 0 || work.ins.empty()) {
    return;
  }
  // Rank-0 inputs (stack of scalars) have an empty shape; rank 0 is the right base.
  work.axis += static_cast<int64_t>(work.ins.front().shape.size()) + behavior->axis_rank_delta;
}

bool BindFinalMask(AiCoreOpWork &work, const LoweringContext &ctx, std::string *err) {
  work.final_mask = ctx.output_final_mask;
  if (work.final_mask.size() == work.dtypes.size()) {
    return true;
  }
  if (err != nullptr) {
    *err = "aicore-binary lowering: final_mask arity " + std::to_string(work.final_mask.size()) +
           " != output arity " + std::to_string(work.dtypes.size()) + " for '" + work.tag + "'";
  }
  return false;
}

void FinalizeAiCoreLoweringResult(const AiCoreOpWork &work, LoweringResult &out) {
  out.outputs = BuildAiCoreOutputTensorRefs(work);
  out.launch = work.stage.LaunchNode();
  out.prepare = work.stage.TilingNode();
  out.mem_to_frees = work.stage.IntermediateAllocs(work.final_mask);
  FillLoweringCniCommon(work.entry, work.ins, work.out_shapes[0], work.dtypes[0], &out);
  // A layout recipe can bind a different OPP schema. OpImpl tiling reads the
  // ComputeNodeInfo attributes, so it must use the same schema as the tiling op description.
  out.cni.op_type = work.bind.opp_type;
  out.cni.opp_input_names = work.bind.opp_input_names != nullptr ? work.bind.opp_input_names : "";
  out.cni.catalog = work.bind.pack_entry;
  out.cni.axis = work.axis;
  out.cni.output_shapes = work.out_shapes;
  out.cni.output_dtypes.clear();
  out.cni.output_dtypes.reserve(work.dtypes.size());
  std::transform(work.dtypes.begin(), work.dtypes.end(), std::back_inserter(out.cni.output_dtypes),
                 [](ge::DataType dt) { return static_cast<int32_t>(dt); });
  out.cni.opp_output_names = (work.entry.opp_output_names != nullptr) ? work.entry.opp_output_names : "";
  out.cni.attrs = work.extras;
  out.cni.compute_nodes = work.stage.ComputeNodes();
}

}  // namespace

const char *AiCoreBinaryOpLowering::Name() const { return "aicore_binary"; }

bool AiCoreBinaryOpLowering::Match(const fxrt::ir::NodePtr &node, const LoweringContext *ctx) const {
  const OpCatalogEntry *entry = LookupOpCatalog(node);
  if (entry == nullptr || entry->lowering != LoweringKind::kAiCoreBinary) {
    return false;
  }
  if (entry->behavior == nullptr || entry->behavior->recipe != InputRecipe::kAsStrided) {
    return true;
  }
  if (node->attrs.count("fxrt_write_dense") != 0U) {
    return true;
  }
  // A compiled-kernel-only as_strided node exists to spell out the layout the kernel already
  // indexes with; materializing would gather a dense buffer the kernel mis-reads. Yield to
  // ViewAlias. The pre-dispatch check (ctx == nullptr) skips this opt-out.
  if (ctx != nullptr && ctx->compiled_kernel_only != nullptr &&
      ctx->compiled_kernel_only->count(node.get()) != 0U) {
    return false;
  }
  // Contiguous as_strided stays ViewAlias; non-contiguous claims Path C
  // (prefix → Slice, else GE AsStrided with storage_offset always wired).
  const bool path_c = IsAsStridedPathC(node);
  return path_c;
}

InputLayoutKind AiCoreBinaryOpLowering::InputLayout(const fxrt::ir::NodePtr &node, size_t) const {
  const auto *entry = LookupOpCatalog(node);
  return entry != nullptr && entry->behavior != nullptr && entry->behavior->recipe == InputRecipe::kAsStrided
             ? InputLayoutKind::kStorageIndexed : InputLayoutKind::kDense;
}

bool AiCoreBinaryOpLowering::Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx,
                                   LoweringResult *out, std::string *err) {
  const OpCatalogEntry *entry = LookupOpCatalog(node);
  return entry != nullptr && LowerAiCoreBinaryWithCatalog(node, *entry, tag, ctx, out, err);
}

bool LowerAiCoreBinaryWithCatalog(const fxrt::ir::NodePtr &node, const OpCatalogEntry &entry,
                                 const std::string &tag, LoweringContext &ctx, LoweringResult *out,
                                 std::string *err) {
  if (out == nullptr || !ValidateLoweringContext(ctx, false, "aicore-binary lowering", err)) {
    return false;
  }
  AiCoreOpWork work{node, entry, tag};
  if (!CollectAiCoreInputs(ctx, work, err) || !RejectPrivateFormat(work, ctx, err)) {
    return false;
  }
  NormalizeCapturedAxis(work);
  if (!ResolveAiCoreOutputs(work, err)) {
    return false;
  }
  work.bind = ResolveAiCoreOppBinding(work);
  if (!BindWorkOptional(work, err) || !SelectAndBuildAiCoreArtifacts(ctx, work, err)) {
    return false;
  }

  const ge::TensorDesc empty_td;
  RtGraphBuilder gb(ctx.main, empty_td, *ctx.sym_gen, ctx.kernel_nodes);
  std::vector<RtGraphBuilder::Value> c_rules;
  RtGraphBuilder::Value c_infer;
  if (!WireAiCoreShapeRuleConsts(work, ctx, gb, c_rules, c_infer, err) || !BindFinalMask(work, ctx, err)) {
    return false;
  }
  if (!ValidateOutputRefs(work, err)) {
    return false;
  }
  WireInferShape(gb, work, c_rules, c_infer, ctx.bind_symbols);
  BindBroadcastTilingViewsToInferShape(ctx, work);
  WireIoAddrs(gb, work, ctx);
  PadAiCoreOptionalInputs(gb, work);
  if (!WireAiCoreExecute(gb, work, *ctx.rt, err)) {
    return false;
  }
  WireAiCoreFinalOutputs(gb, work, *ctx.rt);
  FinalizeAiCoreLoweringResult(work, *out);
  return true;
}

}  // namespace fx_converter

FX_REGISTER_OP_LOWERING(::fx_converter::kAiCoreBinaryPriority,
                        std::make_unique<::fx_converter::AiCoreBinaryOpLowering>());
