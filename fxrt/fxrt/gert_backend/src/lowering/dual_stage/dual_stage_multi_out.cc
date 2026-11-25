#include "lowering/dual_stage/dual_stage_op_lowering.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/runtime/runtime_attrs_def.h"
#include "common/logger.h"
#include "exe_graph/dual_stage_raw_tensor_kernels.h"
#include "exe_graph/es_graph_builder.h"
#include "exe_graph/graph_builder_utils.h"
#include "lowering/catalog_attrs.h"
#include "lowering/common/view_utils.h"
#include "lowering/common/wiring.h"
#include "mrt_ge/mrt_node_utils.h"
#include "register/op_catalog.h"
#include "shape/fx_expr_infer_shape.h"
#include "shape/mrt_to_ge_expr.h"

namespace fx_converter {
namespace {

struct OutSlot {
  std::vector<int64_t> shape;
  ge::DataType dtype = ge::DT_FLOAT;
  bool is_final = false;
  fx_converter::Node *output_data = nullptr;
  fx_converter::Node *alloc = nullptr;
  TensorFieldEndpoints fields;
  fx_converter::Node *boxed = nullptr;
  uint64_t s_infer = 0;
  uint64_t s_addr = 0;
  uint64_t s_output_data = 0;
  uint64_t s_boxed = 0;
};

bool CollectSlots(const fxrt::ir::NodePtr &node, const LoweringContext &ctx, size_t nout, std::vector<OutSlot> *slots,
                  std::string *err) {
  const std::vector<fxrt::ir::TensorPtr> tensors = TensorsOfNode(node);
  if (tensors.size() != nout || ctx.output_final_mask.size() != nout) {
    if (err != nullptr) {
      *err = "dual-stage lowering: multi-output arity does not match the IR tuple";
    }
    return false;
  }
  slots->resize(nout);
  for (size_t i = 0; i < nout; ++i) {
    (*slots)[i].shape = ShapeOfTensor(tensors[i]);
    (*slots)[i].dtype = DtypeOfTensor(tensors[i]);
    (*slots)[i].is_final = ctx.output_final_mask[i];
  }
  return true;
}

fx_converter::Node *AddShapeRule(LoweringContext &ctx, const std::string &tag, size_t index,
                           const fxrt::ir::TensorPtr &tensor, const char *logical_name) {
  std::shared_ptr<FxShapeRule> rule = BuildOutputSymbolicShapeRuleFromTensor(tensor, ctx.symbol_env, logical_name);
  if (rule == nullptr) {
    return nullptr;
  }
  const FxShapeRule *rule_ptr = rule.get();
  ctx.built->shape_rules.push_back(std::move(rule));
  fx_converter::Node *node = AddConstNode(ctx.main, "const_shape_rule_" + tag + "_" + std::to_string(index), &rule_ptr,
                                    sizeof(rule_ptr), false, *ctx.sym_gen, FxrtConstType::kVoidPtr);
  ctx.kernel_nodes->push_back(node);
  return node;
}

bool WireMultiInfer(LoweringContext &ctx, const std::string &tag, const std::vector<TensorRef> &ins,
                    const std::vector<fx_converter::Node *> &rules, const DualStageOpFwk &fwk, std::vector<OutSlot> *slots,
                    fx_converter::Node **infer_out) {
  auto desc = std::make_shared<FxNodeSpec>("infer_" + tag, "InferShape");
  std::vector<uint64_t> in_syms;
  in_syms.reserve(ins.size() + rules.size() + 1U);
  for (const TensorRef &in : ins) {
    desc->AddInputDesc(MakeNdTensorDesc(PhysicalShape(in), in.dtype));
    in_syms.push_back(in.HasUnpackedPair() ? in.unpacked.shape.sym : in.symbol);
  }
  const ge::TensorDesc empty;
  for (fx_converter::Node *rule : rules) {
    desc->AddInputDesc(empty);
    in_syms.push_back(rule->GetExtendInfo()->GetOutputSymbol(0));
  }
  desc->AddInputDesc(empty);
  in_syms.push_back(fwk.s_infer);
  std::vector<uint64_t> out_syms;
  out_syms.reserve(slots->size());
  for (OutSlot &slot : *slots) {
    desc->AddOutputDesc(MakeNdTensorDesc(slot.shape, slot.dtype));
    slot.s_infer = (*ctx.sym_gen)++;
    out_syms.push_back(slot.s_infer);
  }
  fx_converter::Node *infer = AddNode(ctx.main, desc);
  WireSymbols(infer, in_syms, out_syms);
  for (size_t i = 0; i < ins.size(); ++i) {
    const TensorEndpoint shape_ep = ins[i].HasUnpackedPair() ? ins[i].unpacked.shape
                                                              : TensorEndpoint{ins[i].producer, 0, ins[i].symbol};
    if (shape_ep.Valid()) {
      AddDataEdge(ctx.main, shape_ep.node, shape_ep.out, infer, static_cast<int32_t>(i));
    }
  }
  for (size_t i = 0; i < rules.size(); ++i) {
    AddDataEdge(ctx.main, rules[i], 0, infer, static_cast<int32_t>(ins.size() + i));
  }
  AddDataEdge(ctx.main, fwk.c_infer, 0, infer, static_cast<int32_t>(ins.size() + rules.size()));
  if (ctx.bind_symbols != nullptr) {
    AddDataEdge(ctx.main, ctx.bind_symbols, -1, infer, -1);
  }
  ctx.kernel_nodes->push_back(infer);
  *infer_out = infer;
  return true;
}

bool AllocIntermediate(LoweringContext &ctx, const std::string &tag, size_t index, fx_converter::Node *infer, OutSlot *slot) {
  const std::string suffix = tag + "_" + std::to_string(index);
  ge::DataType dtype = slot->dtype;
  fx_converter::Node *dtype_node =
      AddConstNode(ctx.main, "const_out_dtype_" + suffix, &dtype, sizeof(dtype), false, *ctx.sym_gen,
                   FxrtConstType::kGeDataType);
  ctx.kernel_nodes->push_back(dtype_node);
  uint64_t s_size = 0;
  const CalcTensorSizeInputs size_inputs{{dtype_node, 0, dtype_node->GetExtendInfo()->GetOutputSymbol(0)},
                                         {infer, static_cast<int32_t>(index), slot->s_infer}};
  fx_converter::Node *calc = AddCalcTensorSizeNode(ctx.main, suffix, size_inputs, *ctx.sym_gen, &s_size);
  ctx.kernel_nodes->push_back(calc);
  const ge::TensorDesc empty;
  slot->alloc = AddAllocMemoryNode(ctx.main, suffix, empty, ctx.rt->s_allocator, ctx.rt->select_l2, s_size, calc,
                                   *ctx.sym_gen, &slot->s_addr);
  ctx.kernel_nodes->push_back(slot->alloc);
  return slot->alloc != nullptr;
}

bool AllocFinal(LoweringContext &ctx, const std::string &tag, size_t index, fx_converter::Node *infer, bool es_model,
                OutSlot *slot) {
  const std::string suffix = tag + "_" + std::to_string(index);
  const ge::TensorDesc td = MakeNdTensorDesc(slot->shape, slot->dtype);
  const ge::TensorDesc empty;
  slot->output_data = AddOutputDataNode(ctx.main, suffix, td, *ctx.sym_gen, &slot->s_output_data);
  uint64_t s_size = 0;
  fx_converter::Node *size_node = nullptr;
  if (es_model) {
    const size_t capacity = EsModelOutputCapacityBytes(slot->shape, slot->dtype);
    size_node = AddConstNode(ctx.main, "const_out_capacity_" + suffix, &capacity, sizeof(capacity), false,
                             *ctx.sym_gen, FxrtConstType::kSizeT);
    s_size = size_node->GetExtendInfo()->GetOutputSymbol(0);
  } else {
    ge::DataType dtype = slot->dtype;
    fx_converter::Node *dtype_node =
        AddConstNode(ctx.main, "const_out_dtype_" + suffix, &dtype, sizeof(dtype), false, *ctx.sym_gen,
                     FxrtConstType::kGeDataType);
    const CalcTensorSizeInputs size_inputs{{dtype_node, 0, dtype_node->GetExtendInfo()->GetOutputSymbol(0)},
                                           {infer, static_cast<int32_t>(index), slot->s_infer}};
    size_node = AddCalcTensorSizeNode(ctx.main, suffix, size_inputs, *ctx.sym_gen, &s_size);
    ctx.kernel_nodes->push_back(dtype_node);
  }
  slot->alloc = AddAllocModelOutTensorNode(ctx.main, suffix, td, empty, ctx.rt->s_allocator, ctx.rt->select_l2, s_size,
                                           size_node, slot->s_output_data, slot->output_data, *ctx.sym_gen,
                                           &slot->s_addr);
  slot->fields = AddTensorFieldConsts(ctx.main, suffix, slot->dtype, gert::kOnDeviceHbm, ge::FORMAT_ND, *ctx.sym_gen,
                                      ctx.kernel_nodes);
  ctx.kernel_nodes->push_back(slot->output_data);
  ctx.kernel_nodes->push_back(size_node);
  ctx.kernel_nodes->push_back(slot->alloc);
  return slot->alloc != nullptr && slot->fields.Valid();
}

bool BoxSlot(LoweringContext &ctx, const std::string &tag, size_t index, fx_converter::Node *infer, OutSlot *slot) {
  TensorRef ref;
  ref.shape = slot->shape;
  ref.dtype = slot->dtype;
  ref.unpacked.shape = TensorEndpoint{infer, static_cast<int32_t>(index), slot->s_infer};
  ref.unpacked.addr = TensorEndpoint{slot->alloc, 0, slot->s_addr};
  ref.allocation = ref.unpacked.addr;
  if (!EnsureTensorRefBoxed(&ref, ctx.main, tag + "_out" + std::to_string(index), *ctx.sym_gen, ctx.kernel_nodes)) {
    return false;
  }
  slot->boxed = ref.producer;
  slot->s_boxed = ref.symbol;
  return slot->boxed != nullptr;
}

// scale, keep, head, pre, next, inner, sparse, layout. Offsets match the prepare reader.
void PackFlashAttentionAttrs(const CatalogAttrExtras &extras, std::vector<uint8_t> *blob) {
  constexpr size_t kCount = 8U;
  const size_t header = (sizeof(RuntimeAttrsDef) + kCount * sizeof(size_t) + 7U) & ~size_t{7};
  constexpr size_t kValueBytes = 64U;
  blob->assign(header + kValueBytes, 0U);
  auto *attrs_def = reinterpret_cast<RuntimeAttrsDef *>(blob->data());
  attrs_def->attr_num = kCount;
  const size_t slots[] = {0U, 8U, 16U, 24U, 32U, 40U, 48U, 56U};
  for (size_t i = 0; i < kCount; ++i) {
    attrs_def->offset[i] = header + slots[i];
  }
  const float scale = extras.FloatOr("scale_value", 1.f);
  const float keep = extras.FloatOr("keep_prob", 1.f);
  const int64_t head = extras.IntOr("head_num", 0);
  const int64_t pre = extras.IntOr("pre_tockens", 2147483647);
  const int64_t next = extras.IntOr("next_tockens", 2147483647);
  const int64_t inner = extras.IntOr("inner_precise", 0);
  const int64_t sparse = extras.IntOr("sparse_mode", 0);
  char layout[8] = {};
  const char *text = extras.StrOr("input_layout", "BSH");
  if (text == nullptr) {
    text = "BSH";
  }
  std::strncpy(layout, text, sizeof(layout) - 1U);
  uint8_t *base = blob->data();
  std::memcpy(base + header, &scale, sizeof(scale));
  std::memcpy(base + header + 8U, &keep, sizeof(keep));
  std::memcpy(base + header + 16U, &head, sizeof(head));
  std::memcpy(base + header + 24U, &pre, sizeof(pre));
  std::memcpy(base + header + 32U, &next, sizeof(next));
  std::memcpy(base + header + 40U, &inner, sizeof(inner));
  std::memcpy(base + header + 48U, &sparse, sizeof(sparse));
  std::memcpy(base + header + 56U, layout, sizeof(layout));
}

fx_converter::Node *AddLayoutConst(LoweringContext &ctx, const std::string &name, const DualStageTensorLayout &layout,
                             const std::vector<uint8_t> &packed_attrs) {
  DualStageTensorLayout stored = layout;
  stored.packed_attr_bytes = static_cast<uint32_t>(packed_attrs.size());
  std::vector<uint8_t> bytes(sizeof(stored) + packed_attrs.size());
  std::memcpy(bytes.data(), &stored, sizeof(stored));
  if (!packed_attrs.empty()) {
    std::memcpy(bytes.data() + sizeof(stored), packed_attrs.data(), packed_attrs.size());
  }
  fx_converter::Node *node = AddConstNode(ctx.main, name, bytes.data(), bytes.size(), false, *ctx.sym_gen,
                                          FxrtConstType::kBytePtr);
  ctx.kernel_nodes->push_back(node);
  return node;
}

bool WireMultiPrepare(LoweringContext &ctx, const std::string &tag, const std::vector<TensorRef> &ins,
                      const std::vector<OutSlot> &slots, const DualStageOpFwk &fwk,
                      const std::vector<uint8_t> &packed_attrs, fx_converter::Node *infer, fx_converter::Node **prepare_out,
                      uint64_t *s_params, uint64_t *s_ws_size) {
  const ge::TensorDesc empty;
  auto desc = std::make_shared<FxNodeSpec>("prepare_" + tag, "FxExecuteOpPrepare");
  desc->AddInputDesc(empty);
  std::vector<uint64_t> in_syms;
  for (const TensorRef &in : ins) {
    desc->AddInputDesc(MakeNdTensorDesc(PhysicalShape(in), in.dtype));
    in_syms.push_back(in.symbol);
  }
  for (const OutSlot &slot : slots) {
    desc->AddInputDesc(MakeNdTensorDesc(slot.shape, slot.dtype));
    in_syms.push_back(slot.s_boxed);
  }
  desc->AddInputDesc(empty);
  desc->AddInputDesc(empty);
  desc->AddInputDesc(empty);
  in_syms.push_back(ctx.rt->s_options);
  in_syms.push_back(fwk.s_prepare);
  in_syms.push_back(ctx.rt->s_stream);
  desc->AddOutputDesc(empty);
  desc->AddOutputDesc(empty);
  *s_params = (*ctx.sym_gen)++;
  *s_ws_size = (*ctx.sym_gen)++;
  fx_converter::Node *prepare = AddNode(ctx.main, desc);
  DualStageTensorLayout layout;
  layout.tensor_count = static_cast<uint32_t>(ins.size() + slots.size());
  layout.output_count = static_cast<uint32_t>(slots.size());
  fx_converter::Node *layout_node = AddLayoutConst(ctx, "prepare_raw_tensor_layout_" + tag, layout, packed_attrs);
  in_syms.insert(in_syms.begin(), layout_node->GetExtendInfo()->GetOutputSymbol(0));
  WireSymbols(prepare, in_syms, {*s_params, *s_ws_size});
  AddDataEdge(ctx.main, layout_node, 0, prepare, 0);
  for (size_t i = 0; i < ins.size(); ++i) {
    AddDataEdge(ctx.main, ins[i].producer, 0, prepare, static_cast<int32_t>(i + 1U));
  }
  for (size_t i = 0; i < slots.size(); ++i) {
    AddDataEdge(ctx.main, slots[i].boxed, 0, prepare, static_cast<int32_t>(ins.size() + 1U + i));
    AddDataEdge(ctx.main, slots[i].alloc, -1, prepare, -1);
  }
  AddDataEdge(ctx.main, ctx.rt->c_options, 0, prepare, static_cast<int32_t>(ins.size() + slots.size() + 1U));
  AddDataEdge(ctx.main, fwk.c_prepare, 0, prepare, static_cast<int32_t>(ins.size() + slots.size() + 2U));
  AddDataEdge(ctx.main, ctx.rt->split_streams, 0, prepare, static_cast<int32_t>(ins.size() + slots.size() + 3U));
  AddDataEdge(ctx.main, infer, -1, prepare, -1);
  ctx.kernel_nodes->push_back(prepare);
  *prepare_out = prepare;
  return true;
}

bool WireMultiLaunch(LoweringContext &ctx, const std::string &tag, const std::vector<TensorRef> &ins,
                     const std::vector<OutSlot> &slots, const DualStageOpFwk &fwk, fx_converter::Node *prepare,
                     fx_converter::Node *alloc_ws, uint64_t s_params, uint64_t s_ws_addr, uint64_t s_ws_size,
                     fx_converter::Node **launch_out) {
  const ge::TensorDesc empty;
  auto desc = std::make_shared<FxNodeSpec>("launch_" + tag, "FxExecuteOpLaunch");
  desc->AddInputDesc(empty);
  std::vector<uint64_t> in_syms;
  for (const TensorRef &in : ins) {
    desc->AddInputDesc(MakeNdTensorDesc(PhysicalShape(in), in.dtype));
    in_syms.push_back(in.symbol);
  }
  for (const OutSlot &slot : slots) {
    desc->AddInputDesc(MakeNdTensorDesc(slot.shape, slot.dtype));
    in_syms.push_back(slot.s_boxed);
  }
  for (int i = 0; i < 5; ++i) {
    desc->AddInputDesc(empty);
  }
  in_syms.push_back(s_params);
  in_syms.push_back(s_ws_addr);
  in_syms.push_back(s_ws_size);
  in_syms.push_back(ctx.rt->s_stream);
  in_syms.push_back(fwk.s_launch);
  fx_converter::Node *launch = AddNode(ctx.main, desc);
  DualStageTensorLayout layout;
  layout.tensor_count = static_cast<uint32_t>(ins.size() + slots.size());
  layout.output_count = static_cast<uint32_t>(slots.size());
  fx_converter::Node *layout_node = AddLayoutConst(ctx, "launch_raw_tensor_layout_" + tag, layout, {});
  in_syms.insert(in_syms.begin(), layout_node->GetExtendInfo()->GetOutputSymbol(0));
  WireSymbols(launch, in_syms, {});
  const int32_t base = static_cast<int32_t>(ins.size() + slots.size());
  AddDataEdge(ctx.main, layout_node, 0, launch, 0);
  for (size_t i = 0; i < ins.size(); ++i) {
    AddDataEdge(ctx.main, ins[i].producer, 0, launch, static_cast<int32_t>(i + 1U));
  }
  for (size_t i = 0; i < slots.size(); ++i) {
    AddDataEdge(ctx.main, slots[i].boxed, 0, launch, static_cast<int32_t>(ins.size() + 1U + i));
  }
  AddDataEdge(ctx.main, prepare, 0, launch, base + 1);
  AddDataEdge(ctx.main, alloc_ws, 0, launch, base + 2);
  AddDataEdge(ctx.main, prepare, 1, launch, base + 3);
  AddDataEdge(ctx.main, ctx.rt->split_streams, 0, launch, base + 4);
  AddDataEdge(ctx.main, fwk.c_launch, 0, launch, base + 5);
  ctx.kernel_nodes->push_back(launch);
  *launch_out = launch;
  return true;
}

void PublishSlots(const OpCatalogEntry &entry, const std::vector<TensorRef> &ins, const CatalogAttrExtras &extras,
                  const std::vector<OutSlot> &slots, fx_converter::Node *infer, fx_converter::Node *prepare,
                  fx_converter::Node *launch, fx_converter::Node *alloc_ws, LoweringResult *out) {
  out->outputs.resize(slots.size());
  out->cni.compute_nodes.push_back(infer);
  for (size_t i = 0; i < slots.size(); ++i) {
    const OutSlot &slot = slots[i];
    TensorRef &ref = out->outputs[i];
    if (slot.is_final) {
      ref.producer = slot.output_data;
      ref.symbol = slot.s_output_data;
    } else {
      ref.producer = slot.boxed;
      ref.symbol = slot.s_boxed;
      out->mem_to_frees.push_back(slot.alloc);
    }
    ref.shape = slot.shape;
    ref.dtype = slot.dtype;
    ref.unpacked.shape = TensorEndpoint{infer, static_cast<int32_t>(i), slot.s_infer};
    ref.unpacked.addr = TensorEndpoint{slot.alloc, 0, slot.s_addr};
    ref.allocation = ref.unpacked.addr;
    out->cni.compute_nodes.push_back(slot.alloc);
  }
  out->prepare = prepare;
  out->launch = launch;
  out->cni.compute_nodes.push_back(prepare);
  out->cni.compute_nodes.push_back(alloc_ws);
  out->cni.compute_nodes.push_back(launch);
  if (!slots.empty()) {
    FillLoweringCniCommon(entry, ins, slots.front().shape, slots.front().dtype, out);
  }
  out->cni.attrs = extras;
}

}  // namespace

bool LowerMultiOutputDualStage(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx,
                               const OpCatalogEntry &entry, const DualStageOpFwk &fwk, std::vector<TensorRef> *ins,
                               const CatalogAttrExtras &extras, std::string *err, LoweringResult *out) {
  if (ins == nullptr || out == nullptr || ctx.main == nullptr || ctx.sym_gen == nullptr ||
      ctx.kernel_nodes == nullptr || ctx.rt == nullptr) {
    if (err != nullptr) {
      *err = "dual-stage lowering: missing context for multi-output op";
    }
    return false;
  }
  const size_t nout = CatalogOutputArity(entry);
  std::vector<OutSlot> slots;
  if (!CollectSlots(node, ctx, nout, &slots, err)) {
    return false;
  }
  const std::vector<fxrt::ir::TensorPtr> tensors = TensorsOfNode(node);
  std::vector<fx_converter::Node *> rules;
  rules.reserve(nout);
  for (size_t i = 0; i < nout; ++i) {
    fx_converter::Node *rule = AddShapeRule(ctx, tag, i, tensors[i], entry.logical_name);
    if (rule == nullptr) {
      if (err != nullptr) {
        *err = std::string("dual-stage lowering: no shape rule for output ") + std::to_string(i);
      }
      return false;
    }
    rules.push_back(rule);
  }
  fx_converter::Node *infer = nullptr;
  if (!WireMultiInfer(ctx, tag, *ins, rules, fwk, &slots, &infer)) {
    return false;
  }
  const bool es_model = ctx.built != nullptr && ctx.built->fx_builder != nullptr;
  for (size_t i = 0; i < slots.size(); ++i) {
    const bool allocated = slots[i].is_final ? AllocFinal(ctx, tag, i, infer, es_model, &slots[i])
                                             : AllocIntermediate(ctx, tag, i, infer, &slots[i]);
    if (!allocated || !BoxSlot(ctx, tag, i, infer, &slots[i])) {
      if (err != nullptr) {
        *err = "dual-stage lowering: failed to allocate output " + std::to_string(i);
      }
      return false;
    }
  }
  for (size_t i = 0; i < ins->size(); ++i) {
    if (!EnsureTensorRefBoxed(&(*ins)[i], ctx.main, tag + "_in" + std::to_string(i), *ctx.sym_gen, ctx.kernel_nodes)) {
      if (err != nullptr) {
        *err = "dual-stage lowering: failed to box an input";
      }
      return false;
    }
  }
  fx_converter::Node *prepare = nullptr;
  uint64_t s_params = 0;
  uint64_t s_ws_size = 0;
  std::vector<uint8_t> packed_attrs;
  if (entry.opp_type != nullptr && std::strcmp(entry.opp_type, "FlashAttentionScore") == 0) {
    PackFlashAttentionAttrs(extras, &packed_attrs);
  }
  if (!WireMultiPrepare(ctx, tag, *ins, slots, fwk, packed_attrs, infer, &prepare, &s_params, &s_ws_size)) {
    return false;
  }
  const ge::TensorDesc empty;
  auto ws_desc = std::make_shared<FxNodeSpec>("alloc_ws_" + tag, "AllocBatchHbm");
  ws_desc->AddInputDesc(empty);
  ws_desc->AddInputDesc(empty);
  ws_desc->AddOutputDesc(empty);
  fx_converter::Node *alloc_ws = AddNode(ctx.main, ws_desc);
  const uint64_t s_ws_addr = (*ctx.sym_gen)++;
  WireSymbols(alloc_ws, {ctx.rt->s_allocator, s_ws_size}, {s_ws_addr});
  AddDataEdge(ctx.main, ctx.rt->select_l2, 0, alloc_ws, 0);
  AddDataEdge(ctx.main, prepare, 1, alloc_ws, 1);
  ctx.kernel_nodes->push_back(alloc_ws);
  fx_converter::Node *launch = nullptr;
  if (!WireMultiLaunch(ctx, tag, *ins, slots, fwk, prepare, alloc_ws, s_params, s_ws_addr, s_ws_size, &launch)) {
    return false;
  }
  AddFreeBatchHbmNode(ctx.main, tag, empty, s_ws_addr, alloc_ws, launch, ctx.kernel_nodes);
  for (size_t i = 0; i < slots.size(); ++i) {
    if (!slots[i].is_final) {
      continue;
    }
    const EnsureOutMemoryInputs ensure{{infer, static_cast<int32_t>(i), slots[i].s_infer},
                                       {slots[i].alloc, 0, slots[i].s_addr},
                                       slots[i].fields,
                                       {ctx.rt->split_streams, 0, ctx.rt->s_stream},
                                       {slots[i].output_data, 0, slots[i].s_output_data},
                                       {ctx.rt->select_l2, 0, ctx.rt->s_allocator}};
    AddEnsureTensorAtOutMemoryNode(ctx.main, tag + "_" + std::to_string(i), ensure, launch, ctx.kernel_nodes);
  }
  PublishSlots(entry, *ins, extras, slots, infer, prepare, launch, alloc_ws, out);
  return out->prepare != nullptr && out->launch != nullptr;
}

}  // namespace fx_converter
