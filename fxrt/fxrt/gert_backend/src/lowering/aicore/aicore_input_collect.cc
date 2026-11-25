// AICore Path C input collection: walk IR inputs in order and handle each kind as it
// appears (tensor resolve, shape-list materialize at shape_input_index, attr / skip /
// scalar). as_strided has its own walk because its layout lists become prefix Slice or
// GE AsStrided parameters rather than ordinary operator shape inputs.

#include "lowering/aicore/aicore_input_collect.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "exe_graph/execute_graph_builder.h"
#include "ir/value/value.h"
#include "lowering/aicore/aicore_input_materialize.h"
#include "lowering/common/view_utils.h"
#include "lowering/common/wiring.h"
#include "mrt_ge/mrt_node_utils.h"
#include "ops/op_def/ops_name.h"
#include "shape/fx_sym_expr_parser.h"
#include "shape/shape_list_utils.h"

namespace fx_converter {
namespace {

// Row-major contiguous, tolerant of dynamic (wildcard) dims. A dim whose size or
// stride is symbolic cannot disprove contiguity, so it is skipped; expected<0 means
// a dynamic size appeared to the right and the stride this dim would need is unknown.
// Size-1 dims never constrain the layout (their stride is irrelevant, e.g. the
// [B,1,H,D] <-> [B,H,1,D] transpose). Static dims must match exactly.
bool IsRowMajorContiguous(const std::vector<int64_t> &size, const std::vector<int64_t> &stride) {
  if (size.size() != stride.size()) {
    return false;
  }
  int64_t expected = 1;
  for (int64_t i = static_cast<int64_t>(size.size()) - 1; i >= 0; --i) {
    const size_t ui = static_cast<size_t>(i);
    if (size[ui] == 0) {
      return true;
    }
    if (size[ui] != 1 && stride[ui] >= 0 && expected >= 0 && stride[ui] != expected) {
      return false;
    }
    if (size[ui] < 0) {
      expected = -1;
    } else if (expected >= 0) {
      expected *= size[ui];
    }
  }
  return true;
}

// Prefix narrow without trusting IR in_shape (ABI0 stub may be {-1,-1} rank-2).
// Contiguous parent extents implied by stride[i]/stride[i+1]; require at least one narrow.
bool AsStridedStrideImpliesPrefixNarrow(const std::vector<int64_t> &size, const std::vector<int64_t> &stride) {
  if (size.size() != stride.size() || size.empty()) {
    return false;
  }
  const size_t last = size.size() - 1U;
  if (size[last] != 1 && stride[last] != 1) {
    return false;
  }
  bool narrowed = false;
  for (size_t i = 0; i + 1U < size.size(); ++i) {
    if (stride[i + 1U] <= 0 || stride[i] < 0 || stride[i] % stride[i + 1U] != 0) {
      return false;
    }
    const int64_t implied = stride[i] / stride[i + 1U];
    if (size[i + 1U] < 0) {
      continue;  // keep-dim / dynamic: cannot prove this axis is prefix
    }
    if (size[i + 1U] > implied) {
      return false;
    }
    if (size[i + 1U] < implied) {
      narrowed = true;
    }
  }
  return narrowed;
}

// GE/TBE AsStrided always has GM slots x,size,stride,storage_offset (offset optional in proto
// but the .o still expects the 4th pointer), so materialize the offset too, including 0.
// All three carry kGraphIntListDtype: torch sizes/strides/offsets are int64, and 910B ships both
// `/9,2/9,2/` (INT64) and `/3,2/3,2/` (INT32) AsStrided bins, so nothing forces a narrowing here.
// SymInt size, stride and storage_offset entries arrive as -1 and are resolved per Execute by
// the shared shape-payload refresh. A negative literal stride remains invalid; a symbolic slot
// is distinguished by its expression channel and is never launched as the -1 placeholder.
bool CollectAsStridedGeInputs(const LoweringContext &ctx, AiCoreOpWork &work, const std::vector<int64_t> &size,
                              const std::vector<fxrt::ir::SymbolicExprPtr> &size_exprs,
                              const std::vector<int64_t> &stride,
                              const std::vector<fxrt::ir::SymbolicExprPtr> &stride_exprs, int64_t offset,
                              const fxrt::ir::SymbolicExprPtr &offset_expr, std::string *err) {
  if (stride.size() != stride_exprs.size()) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: stride metadata/expression rank mismatch for " + work.tag;
    }
    return false;
  }
  bool negative_stride = false;
  for (size_t i = 0; i < stride.size(); ++i) {
    if (stride[i] < 0 && stride_exprs[i] == nullptr) {
      negative_stride = true;
      break;
    }
  }
  if (negative_stride || (offset < 0 && offset_expr == nullptr)) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: negative literal stride/storage_offset unsupported for " + work.tag;
    }
    return false;
  }
  work.extras.recipe.use_as_strided = true;
  TensorRef size_ref;
  TensorRef stride_ref;
  TensorRef off_ref;
  if (!MaterializeShapeTensor(ctx.main, ctx.built, size, &size_exprs, work.tag, 0, *ctx.sym_gen, ctx.kernel_nodes,
                              false, kGraphIntListDtype, size_ref, err, true)) {
    return false;
  }
  if (!MaterializeShapeTensor(ctx.main, ctx.built, stride, &stride_exprs, work.tag, 1, *ctx.sym_gen,
                              ctx.kernel_nodes, false, kGraphIntListDtype, stride_ref, err, true)) {
    return false;
  }
  const std::vector<fxrt::ir::SymbolicExprPtr> offset_exprs = {offset_expr};
  if (!MaterializeShapeTensor(ctx.main, ctx.built, {offset}, &offset_exprs, work.tag, 2, *ctx.sym_gen,
                              ctx.kernel_nodes, false, kGraphIntListDtype, off_ref, err, true)) {
    return false;
  }
  work.ins.push_back(size_ref);
  work.ins.push_back(stride_ref);
  work.ins.push_back(off_ref);
  return true;
}

size_t ExpectedAiCoreTensorArity(const AiCoreOpWork &work) {
  // arity is the tensor IO count after Collect: scalar-materializing rows
  // (eq_scalar, sub_scalar, ...) declare the trailing scalar's tensor slot in it.
  return work.entry.arity == 0 ? 1 : work.entry.arity;
}

bool MaterializeSliceOffsetsSizeTensors(const LoweringContext &ctx, AiCoreOpWork &work,
                                        const std::vector<int64_t> &offsets, const std::vector<int64_t> &size,
                                        const std::vector<fxrt::ir::SymbolicExprPtr> &size_exprs, std::string *err) {
  const ge::DataType list_dtype = work.entry.shape_list_as_int32 ? ge::DT_INT32 : ge::DT_INT64;
  const bool broadcast_target = work.entry.shape_list_broadcast_target;
  TensorRef off_ref;
  TensorRef size_ref;
  if (!MaterializeShapeTensor(ctx.main, ctx.built, offsets, nullptr, work.tag, 0, *ctx.sym_gen, ctx.kernel_nodes,
                              broadcast_target, list_dtype, off_ref, err)) {
    return false;
  }
  if (!MaterializeShapeTensor(ctx.main, ctx.built, size, &size_exprs, work.tag, 1, *ctx.sym_gen, ctx.kernel_nodes,
                              broadcast_target, list_dtype, size_ref, err)) {
    return false;
  }
  work.ins.push_back(off_ref);
  work.ins.push_back(size_ref);
  const size_t expect = ExpectedAiCoreTensorArity(work);
  if (work.ins.size() != expect) {
    if (err != nullptr) {
      *err = std::string("aicore-binary lowering: arity mismatch for ") + work.entry.logical_name + " expect " +
             std::to_string(expect) + " got " + std::to_string(work.ins.size());
    }
    return false;
  }
  return true;
}

// An omitted as_strided offset means the current source storage origin.  Feed
// TensorV2 carries that origin as a runtime scalar, while the AsStrided .o ABI
// accepts only a materialized offset tensor.  Rebase the host data header and
// clear the runtime field so the existing zero-offset Path C ABI remains valid.
bool RebaseOmittedAsStridedSource(LoweringContext &ctx, AiCoreOpWork &work, std::string *err) {
  if (work.ins.empty() || !work.ins.front().runtime_storage_offset.Valid()) {
    return true;
  }
  auto &source = work.ins.front();
  TensorEndpoint source_addr = source.unpacked.addr;
  TensorEndpoint source_shape = source.unpacked.shape;
  if (!source_addr.Valid() || !source_shape.Valid()) {
    if (source.producer == nullptr) {
      if (err != nullptr) *err = "aicore-binary lowering: omitted as_strided source has no address";
      return false;
    }
    uint64_t shape_symbol = 0;
    uint64_t addr_symbol = 0;
    auto *split = GetOrCreateSplitDataTensor(ctx, source, work.tag + "_offset_source", 0,
                                              &shape_symbol, &addr_symbol);
    if (split == nullptr) {
      if (err != nullptr) *err = "aicore-binary lowering: failed to unpack omitted as_strided source";
      return false;
    }
    source_shape = TensorEndpoint{split, 0, shape_symbol};
    source_addr = TensorEndpoint{split, 1, addr_symbol};
  }
  const int32_t item_size = ge::GetSizeByDataType(source.dtype);
  if (item_size <= 0 || source.storage_offset < 0 ||
      static_cast<uint64_t>(source.storage_offset) >
          std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(item_size)) {
    if (err != nullptr) *err = "aicore-binary lowering: omitted as_strided source offset is invalid";
    return false;
  }
  const size_t byte_offset = static_cast<size_t>(source.storage_offset) * static_cast<size_t>(item_size);
  const size_t element_size = static_cast<size_t>(item_size);
  const ge::TensorDesc empty_td;
  RtGraphBuilder builder(ctx.main, empty_td, *ctx.sym_gen, ctx.kernel_nodes);
  const auto c_byte_offset = builder.Const("const_as_strided_byte_offset_" + work.tag, &byte_offset,
                                           sizeof(byte_offset), false);
  const auto c_item_size = builder.Const("const_as_strided_item_size_" + work.tag, &element_size,
                                         sizeof(element_size), false);
  const auto view = builder.Node("FxContiguousView", "as_strided_source_" + work.tag,
                                 {source_addr, source_shape, source.runtime_storage_offset, c_byte_offset,
                                  c_item_size}, 2U);
  if (!view.outs[0].Valid() || !view.outs[1].Valid()) {
    if (err != nullptr) *err = "aicore-binary lowering: failed to rebase omitted as_strided source";
    return false;
  }
  source.unpacked.addr = view.outs[0];
  source.unpacked.shape = source_shape;
  source.storage_offset = 0;
  source.runtime_storage_offset = {};
  source.allocation = source.allocation.Valid() ? source.allocation : source_addr;
  return true;
}

// Fold a static storage_offset into the data pointer for a generic aicore tensor
// input (the TensorIo recipe: bmm/add/..., which has no as_strided .o ABI slot to
// carry the offset). Without this, a consumer whose device tensor is sized for the
// view reads out of bounds when the offset is non-zero -- e.g. MultiheadAttention's
// v = qkv_clone[2] at element 256 of the 384-wide clone. Addr moves by the offset
// and the offset is cleared; strides are unchanged, so the data read is identical.
bool RebaseStaticOffsetRef(LoweringContext &ctx, TensorRef &ref, const std::string &tag, std::string *err) {
  if (ref.storage_offset == 0 || ref.storage_offset_expr != nullptr || ref.runtime_storage_offset.Valid()) {
    return true;  // nothing static to fold (symbolic/runtime offsets use their own path)
  }
  TensorEndpoint source_addr = ref.unpacked.addr;
  TensorEndpoint source_shape = ref.unpacked.shape;
  if (!source_addr.Valid() || !source_shape.Valid()) {
    if (ref.producer == nullptr) return true;  // leave untouched; downstream validates
    uint64_t shape_symbol = 0;
    uint64_t addr_symbol = 0;
    auto *split = GetOrCreateSplitDataTensor(ctx, ref, tag + "_static_off", 0, &shape_symbol, &addr_symbol);
    if (split == nullptr) {
      if (err != nullptr) *err = "aicore lowering: failed to unpack static-offset source";
      return false;
    }
    source_shape = TensorEndpoint{split, 0, shape_symbol};
    source_addr = TensorEndpoint{split, 1, addr_symbol};
  }
  const int32_t item_size = ge::GetSizeByDataType(ref.dtype);
  if (item_size <= 0 || ref.storage_offset < 0 ||
      static_cast<uint64_t>(ref.storage_offset) >
          std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(item_size)) {
    if (err != nullptr) *err = "aicore lowering: invalid static storage offset";
    return false;
  }
  const size_t byte_offset = static_cast<size_t>(ref.storage_offset) * static_cast<size_t>(item_size);
  const size_t element_size = static_cast<size_t>(item_size);
  const int64_t zero_offset = 0;
  const ge::TensorDesc empty_td;
  RtGraphBuilder builder(ctx.main, empty_td, *ctx.sym_gen, ctx.kernel_nodes);
  const auto c_byte_offset = builder.Const("const_static_byte_offset_" + tag, &byte_offset, sizeof(byte_offset), false);
  const auto c_item_size = builder.Const("const_static_item_size_" + tag, &element_size, sizeof(element_size), false);
  const auto c_zero = builder.Const("const_static_runtime_offset_" + tag, &zero_offset, sizeof(zero_offset), false);
  const auto view = builder.Node("FxContiguousView", "static_offset_source_" + tag,
                                 {source_addr, source_shape, c_zero, c_byte_offset, c_item_size}, 2U);
  if (view.outs.empty() || !view.outs[0].Valid()) {
    if (err != nullptr) *err = "aicore lowering: failed to rebase static-offset source";
    return false;
  }
  ref.allocation = ref.allocation.Valid() ? ref.allocation : source_addr;
  ref.unpacked.addr = view.outs[0];
  ref.unpacked.shape = source_shape;
  ref.storage_offset = 0;
  return true;
}

// Fail loudly when a row declares a reserved role this lowering cannot collect yet.
bool ValidateAiCoreInputRoles(const AiCoreOpWork &work, std::string *err) {
  const OpBehavior *behavior = work.entry.behavior;
  if (behavior == nullptr || behavior->input_roles == nullptr) {
    return true;
  }
  for (uint8_t i = 0; i < behavior->input_role_count; ++i) {
    const InputRole role = behavior->input_roles[i].role;
    if (role != InputRole::kAxisAttr && role != InputRole::kNamedAttr && role != InputRole::kSkip) {
      if (err != nullptr) {
        *err = "aicore-binary lowering: unsupported input role " + std::to_string(static_cast<int>(role)) + " for " +
               work.tag;
      }
      return false;
    }
  }
  return true;
}

const InputRoleRule *TakeNextNamedAttr(const OpBehavior *behavior, IrInputKind kind, std::vector<uint8_t> &used) {
  if (behavior == nullptr || behavior->input_roles == nullptr) {
    return nullptr;
  }
  for (uint8_t i = 0; i < behavior->input_role_count; ++i) {
    const InputRoleRule &rule = behavior->input_roles[i];
    if (rule.match != kind || rule.role != InputRole::kNamedAttr) {
      continue;
    }
    if (i < used.size() && used[i] != 0) {
      continue;
    }
    if (i < used.size()) {
      used[i] = 1;
    }
    return &rule;
  }
  return nullptr;
}

void StoreNamedFromIr(const fxrt::ir::Value &out, const char *name, CatalogAttrExtras &extras) {
  if (name == nullptr) {
    return;
  }
  if (out.IsBool()) {
    extras.SetBool(name, out.ToBool());
  } else if (out.IsInt()) {
    extras.SetInt(name, out.ToInt());
  } else if (out.IsString()) {
    // The scan buckets strings under kDouble (IrInputKind has no kStr); the captured
    // text borrows the IR storage, valid for the convert scope (CrossEntropyLoss
    // reduction="none" selects the kernel variant, so the default cannot ride).
    extras.SetStr(name, out.ToString().c_str());
  } else {
    extras.SetDouble(name, out.ToDouble());
  }
}

// One collected scalar operand in scan order: a literal (expr=nullptr) or a
// Dynamo-specialized SymInt folded back to a literal, or a genuinely symbolic scalar
// (expr set, value a placeholder). Literal, folded and symbolic slots share one ordered
// list so mixed signatures (arange [start, limit=s27, delta]) keep the proto order.
struct FoldedScalar {
  fxrt::ir::ValuePtr value;
  // Non-null when the scalar is genuinely symbolic (a runtime-resolved SymInt placeholder):
  // the materialized payload is registered for a per-Execute refresh against this expression.
  fxrt::ir::SymbolicExprPtr expr;
};

// Graph rewrite already supplied one integer tuple per missing tensor input.
bool TuplesFillMissingTensorSlots(const AiCoreOpWork &work, const std::vector<fxrt::ir::NodePtr> &shape_list_nodes,
                                  const std::vector<FoldedScalar> &scalars) {
  return !shape_list_nodes.empty() && scalars.empty() &&
         work.ins.size() + shape_list_nodes.size() == ExpectedAiCoreTensorArity(work);
}

// A second axis candidate means the IR carried two role-ambiguous ints (e.g. argsort with
// descending=true serialized as Int 1 next to a positive axis value). Refuse to guess which
// one is the axis -- a silently overwritten axis picks the wrong kernel variant.
bool AssignAxisOnce(int64_t value, AiCoreOpWork &work, std::string *err) {
  if (work.axis != std::numeric_limits<int64_t>::min()) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: multiple axis candidates for " + work.tag +
             "; an Int attr (e.g. descending serialized as 1) is indistinguishable from a "
             "positive axis value, keep such attrs boolean in the traced body";
    }
    return false;
  }
  work.axis = value;
  return true;
}

// A Dynamo-specialized SymInt (Sym(N), e.g. a parametrize-injected axis under dynamic=True)
// is a constant expression, not a free symbol. Fold it back to the literal axis, the same
// fold ReadIntListWithSymbols applies to shape lists. A genuinely symbolic scalar is a
// runtime-resolved value (a Dynamo-lifted SymInt placeholder). When the row bakes it into an
// attr, the value selects the kernel variant at Convert and cannot follow a runtime value.
bool ScanSymbolicScalarInput(const fxrt::ir::NodePtr &in_node, LoweringContext &ctx, AiCoreOpWork &work,
                             std::vector<FoldedScalar> &scalars,
                             std::array<uint8_t, kIrInputKindCount> &kind_ordinals, std::string *err) {
  const OpBehavior *behavior = work.entry.behavior;
  const auto &out = in_node->output;
  // A SymInt slot takes one Int ordinal like a literal Int does, so a SkipIntAt(n) row
  // matches the same argument whether a sibling scalar arrived literal (static trace)
  // or symbolic (dynamic trace) -- arange [start, limit, delta, dtype] needs the dtype
  // skipped at ordinal 3 in both forms.
  ++kind_ordinals[static_cast<size_t>(IrInputKind::kInt)];
  const auto expr = ParseFxSymExpr(out->ToSymbol(), *ctx.symbol_env);
  if (expr != nullptr && expr->GetKind() == fxrt::ir::SymbolicExpr::Kind::Constant) {
    const int64_t folded = expr->Evaluate();
    if (HasInputRole(behavior, IrInputKind::kInt, InputRole::kAxisAttr)) {
      return AssignAxisOnce(folded, work, err);
    }
    // A non-axis scalar symbol (e.g. a cumsum axis materialized as a rank-0 tensor input)
    // still needs a device scalar tensor; queue the folded literal so the collect step
    // materializes it instead of the symbol node.
    scalars.push_back({fxrt::ir::MakeIntrusive<fxrt::ir::Value>(folded), nullptr});
    return true;
  }
  if (expr == nullptr) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: unparsable symbolic scalar input for " + work.tag;
    }
    return false;
  }
  // CaptureAxis / CaptureFill bake the value into an attr. Otherwise the collect step
  // materializes a rank-0 payload whose value RefreshAiCoreShapeTensors rewrites per
  // Execute (ArgMaxV2 dimension / GatherV2 axis / Cumsum axis / mul_scalar operand).
  if (HasInputRole(behavior, IrInputKind::kInt, InputRole::kAxisAttr)) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: symbolic axis for " + work.tag +
             " is baked into an attr at Convert and cannot follow a runtime SymInt; keep the axis a "
             "literal in the traced body, compile with dynamic=False, or mark the dim static";
    }
    return false;
  }
  if (HasInputRole(behavior, IrInputKind::kInt, InputRole::kNamedAttr) ||
      HasInputRole(behavior, IrInputKind::kDouble, InputRole::kNamedAttr)) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: symbolic scalar for " + work.tag +
             " is captured into a GE attr at Convert and cannot follow a runtime SymInt; keep it a "
             "literal in the traced body, compile with dynamic=False, or mark it static";
    }
    return false;
  }
  scalars.push_back({fxrt::ir::MakeIntrusive<fxrt::ir::Value>(int64_t{0}), expr});
  return true;
}

// Declared roles classify a literal non-tensor input; undeclared kinds keep the defaults
// (bool ignored, int/double materialized as a scalar tensor).
bool ScanLiteralNonTensorInput(const fxrt::ir::NodePtr &in_node, AiCoreOpWork &work,
                               std::vector<FoldedScalar> &scalars, std::vector<uint8_t> &used_roles,
                               std::array<uint8_t, kIrInputKindCount> &kind_ordinals, std::string *err) {
  const OpBehavior *behavior = work.entry.behavior;
  const auto &out = in_node->output;
  const IrInputKind kind = out->IsBool() ? IrInputKind::kBool
                                         : (out->IsInt() ? IrInputKind::kInt : IrInputKind::kDouble);
  // Literal non-tensor inputs of one kind are numbered in IR order so a rule can skip
  // only the n-th (SkipIntAt); the symbol channel takes an Int ordinal too, so a row
  // matches the same slot in static and dynamic traces.
  const int64_t ordinal = kind_ordinals[static_cast<size_t>(kind)]++;
  if (FindInputRoleAt(behavior, kind, InputRole::kSkip, ordinal) != nullptr) {
    return true;
  }
  if (HasInputRole(behavior, kind, InputRole::kNamedAttr)) {
    const InputRoleRule *rule = TakeNextNamedAttr(behavior, kind, used_roles);
    if (rule != nullptr) {
      StoreNamedFromIr(*out, rule->attr_name, work.extras);
    }
    return true;
  }
  if (out->IsInt() && HasInputRole(behavior, IrInputKind::kInt, InputRole::kAxisAttr)) {
    return AssignAxisOnce(out->ToInt(), work, err);
  }
  if (out->IsDouble() || out->IsInt()) {
    // One ordered scalar list for literal and symbolic slots alike: arange mixes them
    // ([start, limit=s27, delta]) and the proto input order must survive the split.
    scalars.push_back({in_node->output, nullptr});
  }
  return true;
}

bool ScanScalarOrAttrInput(const fxrt::ir::NodePtr &in_node, LoweringContext &ctx, AiCoreOpWork &work,
                           std::vector<FoldedScalar> &scalars, std::vector<uint8_t> &used_roles,
                           std::array<uint8_t, kIrInputKindCount> &kind_ordinals, std::string *err) {
  if (in_node->output->IsSymbol()) {
    return ScanSymbolicScalarInput(in_node, ctx, work, scalars, kind_ordinals, err);
  }
  return ScanLiteralNonTensorInput(in_node, work, scalars, used_roles, kind_ordinals, err);
}

bool ReadAsStridedLayoutArguments(const std::vector<fxrt::ir::NodePtr> &shape_list_nodes,
                                  const std::vector<FoldedScalar> &scalars, const LoweringContext &ctx,
                                  const AiCoreOpWork &work, std::vector<int64_t> *size,
                                  std::vector<fxrt::ir::SymbolicExprPtr> *size_exprs, std::vector<int64_t> *stride,
                                  std::vector<fxrt::ir::SymbolicExprPtr> *stride_exprs, int64_t *offset,
                                  fxrt::ir::SymbolicExprPtr *offset_expr, std::string *err) {
  if (size == nullptr || size_exprs == nullptr || stride == nullptr || stride_exprs == nullptr || offset == nullptr ||
      offset_expr == nullptr) {
    return false;
  }
  *offset = 0;
  *offset_expr = nullptr;
  if (shape_list_nodes.size() == 1U && scalars.empty()) {
    std::vector<int64_t> packed;
    std::vector<fxrt::ir::SymbolicExprPtr> packed_exprs;
    if (shape_list_nodes[0] == nullptr || shape_list_nodes[0]->output == nullptr ||
        !shape_list_nodes[0]->output->IsTuple() ||
        !ReadIntListWithSymbols(shape_list_nodes[0]->output->ToTuple(), *ctx.symbol_env, &packed, &packed_exprs)) {
      if (err != nullptr) {
        *err = "aicore-binary lowering: invalid packed as_strided layout for " + work.tag;
      }
      return false;
    }
    if (packed.empty() || packed[0] < 0 || packed_exprs.empty() || packed_exprs[0] != nullptr ||
        packed.size() != static_cast<size_t>(1 + 2 * packed[0] + 1) ||
        !ParseAsStridedPackedLayout(packed, "aicore-binary lowering", size, stride, offset, err)) {
      return false;
    }
    const size_t rank = size->size();
    size_exprs->assign(packed_exprs.begin() + 1, packed_exprs.begin() + 1 + static_cast<std::ptrdiff_t>(rank));
    stride_exprs->assign(packed_exprs.begin() + static_cast<std::ptrdiff_t>(1U + rank),
                         packed_exprs.begin() + static_cast<std::ptrdiff_t>(1U + 2U * rank));
    *offset_expr = packed_exprs[1U + 2U * rank];
    return true;
  }
  if (shape_list_nodes.size() != 2U || scalars.size() > 1U || shape_list_nodes[0] == nullptr ||
      shape_list_nodes[1] == nullptr || shape_list_nodes[0]->output == nullptr ||
      shape_list_nodes[1]->output == nullptr || !shape_list_nodes[0]->output->IsTuple() ||
      !shape_list_nodes[1]->output->IsTuple() ||
      !ReadIntListWithSymbols(shape_list_nodes[0]->output->ToTuple(), *ctx.symbol_env, size, size_exprs) ||
      !ReadIntListWithSymbols(shape_list_nodes[1]->output->ToTuple(), *ctx.symbol_env, stride, stride_exprs) ||
      size->size() != stride->size()) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: expected separate size/stride lists for " + work.tag;
    }
    return false;
  }
  if (!scalars.empty()) {
    const FoldedScalar &scalar = scalars.front();
    *offset_expr = scalar.expr;
    if (scalar.value != nullptr && scalar.value->IsInt()) {
      *offset = scalar.value->ToInt();
    } else if (scalar.expr == nullptr) {
      if (err != nullptr) {
        *err = "aicore-binary lowering: non-integer storage_offset for " + work.tag;
      }
      return false;
    }
  }
  return true;
}

// A constant size that just repeats the parent's dim0 pins the batch to whatever the traced
// example had. Take the parent's own symbol for that axis instead -- the graph knows it, and
// this used to become a bare -1 for Execute to guess a batch for.
void AdoptParentBatchSymbol(const fxrt::ir::NodePtr &data_node, const LoweringContext &ctx,
                            const std::vector<int64_t> &in_shape, std::vector<int64_t> &size,
                            std::vector<fxrt::ir::SymbolicExprPtr> &size_exprs) {
  if (in_shape.size() != size.size() || size.front() <= 0 || size.front() != in_shape.front()) {
    return;
  }
  std::vector<fxrt::ir::SymbolicExprPtr> data_exprs;
  if (!SymbolicShapeExprsOfNode(data_node, *ctx.symbol_env, &data_exprs) || data_exprs.size() != size.size() ||
      data_exprs.front() == nullptr) {
    return;
  }
  size.front() = kSymbolicDim;
  size_exprs.front() = data_exprs.front();
}

fxrt::ir::NodePtr AiCoreDataNode(const fxrt::ir::NodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    const auto &in = node->inputs[i];
    if (in != nullptr && in->output != nullptr && in->output->IsTensor()) {
      return in;
    }
  }
  return nullptr;
}

// Recover an ordinary unit-step slice from concrete storage geometry. Verify
// the actual storage strides and every coordinate, including the origin.
bool StaticSliceOffsets(const std::vector<int64_t> &physical, const std::vector<int64_t> &size,
                        const std::vector<int64_t> &stride, int64_t offset,
                        std::vector<int64_t> *begin) {
  if (physical.size() != size.size() || size.size() != stride.size() || size.empty() || offset < 0) {
    return false;
  }
  std::vector<int64_t> base_stride;
  if (!ContiguousRowMajorStrides(physical, &base_stride)) return false;
  for (size_t i = 0; i < size.size(); ++i) {
    if (physical[i] <= 0 || size[i] <= 0 || base_stride[i] <= 0 || stride[i] != base_stride[i]) return false;
    const int64_t first = offset / base_stride[i];
    offset %= base_stride[i];
    if (first >= physical[i] || size[i] > physical[i] - first) return false;
    begin->push_back(first);
  }
  return offset == 0;
}

// Choose Slice only after proving its coordinate geometry;
// general symbolic, overlapping or rank-changing layouts retain AsStrided.
bool CollectAsStridedPathCInputs(LoweringContext &ctx, AiCoreOpWork &work,
                                 const std::vector<fxrt::ir::NodePtr> &shape_list_nodes,
                                 const std::vector<FoldedScalar> &scalars, std::string *err) {
  if (work.ins.empty()) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: as_strided missing data tensor for " + work.tag;
    }
    return false;
  }
  // Address the storage the size / stride / offset were computed against. The walk
  // wired whatever tensor the IR input names, which for a view the converter materialized
  // (t()/permute -> Transpose, expand -> BroadcastTo) is the view applied once already;
  // slicing or striding that buffer would apply it a second time.
  fxrt::ir::NodePtr data_node;
  if (!ResolveAsStridedBaseTensor(work.node, ctx, &work.ins[0], &data_node, err)) {
    return false;
  }
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  std::vector<fxrt::ir::SymbolicExprPtr> size_exprs;
  std::vector<fxrt::ir::SymbolicExprPtr> stride_exprs;
  fxrt::ir::SymbolicExprPtr offset_expr;
  int64_t offset = 0;
  if (!ReadAsStridedLayoutArguments(shape_list_nodes, scalars, ctx, work, &size, &size_exprs, &stride, &stride_exprs,
                                    &offset, &offset_expr, err)) {
    return false;
  }
  // The Python hook uses -1 only for an omitted storage_offset. It is not a
  // negative runtime offset. Resolve the omitted form against the source
  // layout; a per-execute source offset still needs a dedicated scalar ABI.
  if (offset == -1 && offset_expr == nullptr) {
    if (work.ins.front().runtime_storage_offset.Valid()) {
      if (!RebaseOmittedAsStridedSource(ctx, work, err)) return false;
    }
    offset = work.ins.front().storage_offset;
  }
  std::vector<int64_t> physical_stride;
  // Stride ratios alone cannot distinguish narrowing from a stepped slice:
  // [8,6,16]/[192,32,1] is not a prefix of [8,12,16]. Require the actual
  // base layout to prove the Slice specialization; otherwise use AsStrided.
  const bool matches_base = !work.ins.empty() &&
      ContiguousRowMajorStrides(PhysicalShape(work.ins[0]), &physical_stride) && physical_stride == stride;
  if (matches_base && AsStridedStrideImpliesPrefixNarrow(size, stride) && offset == 0 && offset_expr == nullptr) {
    std::vector<int64_t> offsets(size.size(), 0);
    AdoptParentBatchSymbol(data_node, ctx, work.ins[0].shape, size, size_exprs);
    return MaterializeSliceOffsetsSizeTensors(ctx, work, offsets, size, size_exprs, err);
  }
  std::vector<int64_t> begin;
  if (offset_expr == nullptr &&
      std::none_of(size_exprs.begin(), size_exprs.end(), [](const auto &expr) { return expr != nullptr; }) &&
      std::none_of(stride_exprs.begin(), stride_exprs.end(), [](const auto &expr) { return expr != nullptr; }) &&
      StaticSliceOffsets(PhysicalShape(work.ins[0]), size, stride, offset, &begin)) {
    return MaterializeSliceOffsetsSizeTensors(ctx, work, begin, size, size_exprs, err);
  }
  if (!CollectAsStridedGeInputs(ctx, work, size, size_exprs, stride, stride_exprs, offset, offset_expr, err)) {
    return false;
  }
  if (work.ins.size() != kAsStridedLaunchInputCount) {
    if (err != nullptr) {
      *err = std::string("aicore-binary lowering: AsStrided arity expect ") +
             std::to_string(kAsStridedLaunchInputCount) + " got " + std::to_string(work.ins.size()) + " for " +
             work.tag;
    }
    return false;
  }
  return true;
}

// Shape-list (Expand / Tile / TopKV2 k) + scalars + cmp scalar synthesis.
void RestoreLeadingKeepDimPads(const std::vector<int64_t> &orig, size_t data_rank, std::vector<int64_t> &dims) {
  if (dims.size() <= data_rank) {
    return;
  }
  const size_t extra = dims.size() - data_rank;
  for (size_t pad_i = 0; pad_i < extra; ++pad_i) {
    if (orig[pad_i] == -1 || orig[pad_i] == std::numeric_limits<int64_t>::min()) {
      dims[pad_i] = orig[pad_i];
    }
  }
}

bool HasInnerKeepDim(const std::vector<int64_t> &dims) {
  if (dims.size() < 2U) {
    return false;
  }
  return std::any_of(dims.begin() + 1, dims.end(), [](int64_t d) {
    return d == -1 || d == std::numeric_limits<int64_t>::min();
  });
}

std::vector<int64_t> PrepareShapeListDims(const AiCoreOpWork &work, std::vector<int64_t> dims,
                                          const std::vector<int64_t> &op_out_shape) {
  if (work.entry.shape_list_broadcast_target && !work.ins.empty()) {
    const std::vector<int64_t> orig = dims;
    const std::vector<int64_t> &data_shape = work.ins[0].shape;
    // Inner keep-dims must be baked from the Expand IR output ([s0, 8, s53]), not
    // the data TensorRef. unsqueeze makes data [s0, 8, 1]; resolving [-1,-1,-1]
    // against that fills the last keep-dim with 1, HasInnerKeepDim becomes false,
    // and Execute never recovers 128 from the embedding's last dim.
    // Check orig (not the post-data list): once a keep-dim is replaced by 1 it
    // is no longer a keep-dim and cannot be overwritten by the IR output.
    if (HasInnerKeepDim(orig) && !op_out_shape.empty()) {
      (void)ResolveKeepDimShapeList(&dims, op_out_shape);
    } else {
      (void)ResolveKeepDimShapeList(&dims, data_shape);
      // Leading 1s from right-align padding are not real keep-dim values; leave -1 for Execute.
      RestoreLeadingKeepDimPads(orig, data_shape.size(), dims);
    }
  }
  return dims;
}

// Reduce-style axes wrap negative literals by the input rank; the reduce bins miscompute
// (or crash) on a literal -1 axis. Symbolic elements keep their expression for the
// Execute-time payload refresh, so only rewrite literals.
void NormalizeShapeListAxes(int64_t rank, std::vector<int64_t> &raw,
                            const std::vector<fxrt::ir::SymbolicExprPtr> &exprs) {
  for (size_t d = 0; d < raw.size(); ++d) {
    const bool is_symbol = d < exprs.size() && exprs[d] != nullptr;
    if (!is_symbol && raw[d] < 0) {
      raw[d] += rank;
    }
  }
}

// Expressions stay meaningful only while the list keeps its shape-list identity and length.
// A baked number drops its symbol. A literal -1 keep-dim takes the op output's own symbol.
void SyncShapeListExprs(const std::vector<int64_t> &dims,
                        const std::vector<fxrt::ir::SymbolicExprPtr> &op_out_exprs,
                        std::vector<fxrt::ir::SymbolicExprPtr> &exprs) {
  if (dims.size() != exprs.size()) {
    exprs.clear();
    return;
  }
  for (size_t d = 0; d < dims.size(); ++d) {
    if (dims[d] != kSymbolicDim) {
      exprs[d] = nullptr;
    } else if (d < op_out_exprs.size() && exprs[d] == nullptr) {
      exprs[d] = op_out_exprs[d];
    }
  }
}

bool PlaceOneShapeList(const LoweringContext &ctx, AiCoreOpWork &work, const fxrt::ir::NodePtr &list_node,
                       size_t &lists_placed, std::string *err) {
  if (list_node == nullptr || list_node->output == nullptr) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: null shape tuple for " + work.tag;
    }
    return false;
  }
  const auto &tup = list_node->output->ToTuple();
  if (tup == nullptr) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: null shape tuple for " + work.tag;
    }
    return false;
  }
  std::vector<fxrt::ir::SymbolicExprPtr> op_out_exprs;
  (void)SymbolicShapeExprsOfNode(work.node, *ctx.symbol_env, &op_out_exprs);
  const std::vector<int64_t> op_out_shape = ShapeOfNode(work.node);
  const bool broadcast_target = work.entry.shape_list_broadcast_target;
  const ge::DataType list_dtype = work.entry.shape_list_as_int32 ? ge::DT_INT32 : ge::DT_INT64;
  const bool defer_tiling_view = broadcast_target && work.entry.shape_input_index >= 0;
  TensorRef ref;
  std::vector<int64_t> raw;
  std::vector<fxrt::ir::SymbolicExprPtr> exprs;
  if (!ReadIntListWithSymbols(tup, *ctx.symbol_env, &raw, &exprs)) {
    if (err != nullptr) {
      *err = "aicore-binary lowering: unparsable symbolic dim in shape list for " + work.tag;
    }
    return false;
  }
  if (work.entry.shape_list_axes_normalize && !work.ins.empty()) {
    const int64_t rank = static_cast<int64_t>(work.ins.front().shape.size());
    NormalizeShapeListAxes(rank, raw, exprs);
  }
  const std::vector<int64_t> dims = PrepareShapeListDims(work, raw, op_out_shape);
  SyncShapeListExprs(dims, op_out_exprs, exprs);
  if (!MaterializeShapeTensor(ctx.main, ctx.built, dims, &exprs, work.tag, lists_placed, *ctx.sym_gen,
                              ctx.kernel_nodes, broadcast_target, list_dtype, ref, err, false, defer_tiling_view)) {
    return false;
  }
  size_t dest = work.ins.size();
  if (work.entry.shape_input_index >= 0) {
    dest = static_cast<size_t>(work.entry.shape_input_index) + lists_placed;
    if (dest > work.ins.size()) {
      dest = work.ins.size();
    }
  }
  work.ins.insert(work.ins.begin() + static_cast<ptrdiff_t>(dest), ref);
  ++lists_placed;
  return true;
}

ge::DataType CollectedScalarDtype(const AiCoreOpWork &work) {
  if (work.entry.scalar_dtype >= 0) {
    return static_cast<ge::DataType>(work.entry.scalar_dtype);
  }
  // Fill's first input describes dimensions (int64), not the value dtype.
  if (work.entry.opp_type != nullptr && std::string(work.entry.opp_type) == "Fill") {
    return DtypeOfNode(work.node);
  }
  // Scalar-only generators (Range) have no leading tensor from which to
  // inherit dtype. Their output descriptor supplies the scalar type.
  return work.ins.empty() ? DtypeOfNode(work.node) : work.ins[0].dtype;
}

bool FlushPendingScalars(const LoweringContext &ctx, AiCoreOpWork &work, const std::vector<FoldedScalar> &scalars,
                         size_t &done, size_t &scalar_idx, std::string *err) {
  const ge::DataType scalar_dtype = CollectedScalarDtype(work);
  for (; done < scalars.size(); ++done) {
    TensorRef ref;
    void *scalar_host = nullptr;
    void *scalar_dev = nullptr;
    const FoldedScalar &scalar = scalars[done];
    const bool symbolic = (scalar.expr != nullptr);
    if (!MaterializeScalarTensor(ctx.main, ctx.built, *scalar.value, scalar_dtype, work.tag, scalar_idx, *ctx.sym_gen,
                                 ctx.kernel_nodes, ref, err, symbolic ? &scalar_host : nullptr,
                                 symbolic ? &scalar_dev : nullptr)) {
      return false;
    }
    if (symbolic) {
      BuiltExecuteGraph::AiCoreShapeTensorRefresh refresh;
      refresh.host = scalar_host;
      refresh.dev = scalar_dev;
      refresh.size_template = {0};
      refresh.payload_type = scalar_dtype;
      refresh.axis_exprs = {scalar.expr};
      refresh.allow_nonpositive = true;
      ctx.built->aicore_shape_refreshes.push_back(std::move(refresh));
    }
    work.ins.push_back(ref);
    ++scalar_idx;
  }
  return true;
}

// GatherV2 dim0 axis is not an IR input. A length-1 INT32 list is a rank-0 scalar
// unless force_vector keeps StorageShape [1], which is what the A5 bins pack.
bool AppendDim0Axis(const LoweringContext &ctx, AiCoreOpWork &work, std::string *err) {
  if (!work.entry.append_dim0_axis) {
    return true;
  }
  TensorRef axis_ref;
  if (!MaterializeShapeTensor(ctx.main, ctx.built, {int64_t{0}}, nullptr, work.tag, work.ins.size(), *ctx.sym_gen,
                              ctx.kernel_nodes, false, ge::DT_INT32, axis_ref, err, true)) {
    return false;
  }
  work.ins.push_back(std::move(axis_ref));
  return true;
}

bool CheckCollectedArity(const AiCoreOpWork &work, std::string *err) {
  const size_t expect = ExpectedAiCoreTensorArity(work);
  const bool arity_ok = (work.entry.arity == 0)
                            ? !work.ins.empty()
                            : (work.ins.size() >= expect &&
                               work.ins.size() <= expect + work.optional.optional_total);
  if (arity_ok) {
    return true;
  }
  if (err != nullptr) {
    *err = std::string("aicore-binary lowering: arity mismatch for ") + work.entry.logical_name + " expect " +
           std::to_string(expect) + " got " + std::to_string(work.ins.size());
  }
  return false;
}

}  // namespace

bool IsAsStridedPathC(const fxrt::ir::NodePtr &node) {
  if (node == nullptr || node->op != fxrt::ops::Op_as_strided_view || node->inputs.empty()) {
    return false;
  }
  // The converter may keep size, stride and storage_offset as separate IR inputs
  // rather than the legacy packed tuple. Do not let ViewAlias treat a symbolic
  // offset as zero merely because it is outside the tuple.
  for (size_t i = 1; i < node->inputs.size(); ++i) {
    const auto &in = node->inputs[i];
    if (in == nullptr || in->output == nullptr) {
      continue;
    }
    if (in->output->IsSymbol()) {
      return true;
    }
    if (in->output->IsTuple() &&
        std::any_of(in->output->ToTuple()->begin(), in->output->ToTuple()->end(),
                    [](const fxrt::ir::ValuePtr &value) { return value != nullptr && value->IsSymbol(); })) {
      return true;
    }
  }
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  int64_t offset = 0;
  if (!TryReadAsStridedLayout(node, &size, &stride, &offset)) {
    return false;
  }
  // A symbolic stride or storage offset cannot be classified from the compile-time
  // sample. Route it to Path C so the packed expressions can be evaluated per Execute
  // and supplied to the AsStrided kernel instead of rejecting the node in ViewAlias.
  for (size_t i = 1; i < node->inputs.size(); ++i) {
    const auto &in = node->inputs[i];
    if (in == nullptr || in->output == nullptr || !in->output->IsTuple()) {
      continue;
    }
    const auto &tuple = *in->output->ToTuple();
    if (tuple.Size() == 0 || tuple[0] == nullptr || !tuple[0]->IsInt()) {
      continue;
    }
    const int64_t rank = tuple[0]->ToInt();
    if (rank < 0 || tuple.Size() < static_cast<size_t>(1 + 2 * rank + 1)) {
      continue;
    }
    const size_t stride_begin = static_cast<size_t>(1 + rank);
    const size_t offset_index = stride_begin + static_cast<size_t>(rank);
    const bool symbolic_stride = std::any_of(
        tuple.begin() + static_cast<std::ptrdiff_t>(stride_begin),
        tuple.begin() + static_cast<std::ptrdiff_t>(offset_index), [](const auto &element) {
          return element != nullptr && element->IsSymbol();
        });
    if (symbolic_stride || (tuple[offset_index] != nullptr && tuple[offset_index]->IsSymbol())) {
      return true;
    }
  }
  // offset!=0 cannot stay a metadata view; Path C AsStrided copies.
  if (offset != 0) {
    return true;
  }
  // ViewAlias keeps validated expand / permute geometry. Generic zero-stride
  // overlapping layouts stay on Path C. Dynamic-dim contiguous still uses the
  // local row-major check (more lenient than IsContiguousRowMajor).
  if (node->inputs[0] == nullptr) {
    return !IsRowMajorContiguous(size, stride);
  }
  const LayoutKind kind = ClassifyAsStridedLayout(ShapeOfNode(node->inputs[0]), size, stride, offset);
  if (kind == LayoutKind::kBroadcast || kind == LayoutKind::kPermute) {
    return false;
  }
  return !IsRowMajorContiguous(size, stride);
}

bool PushResolvedTensor(const fxrt::ir::NodePtr &in_node, const LoweringContext &ctx, AiCoreOpWork &work,
                        std::string *err) {
  TensorRef ref;
  if (!ResolveProducerTensor(in_node, ctx, &ref, err)) {
    return false;
  }
  work.ins.push_back(ref);
  return true;
}

// TensorIo inputs carry no as_strided offset slot, so fold any static offset into
// the data pointer before the kernel consumes it (see RebaseStaticOffsetRef).
bool PushResolvedTensorRebased(const fxrt::ir::NodePtr &in_node, LoweringContext &ctx, AiCoreOpWork &work,
                               std::string *err) {
  TensorRef ref;
  if (!ResolveProducerTensor(in_node, ctx, &ref, err)) {
    return false;
  }
  const std::string tag = work.tag + "_in" + std::to_string(work.ins.size());
  if (!RebaseStaticOffsetRef(ctx, ref, tag, err)) {
    return false;
  }
  work.ins.push_back(ref);
  return true;
}

bool CollectAiCoreTensorIoInputs(LoweringContext &ctx, AiCoreOpWork &work, std::string *err) {
  std::vector<FoldedScalar> scalars;
  const OpBehavior *behavior = work.entry.behavior;
  const uint8_t role_count = (behavior != nullptr) ? behavior->input_role_count : 0;
  std::vector<uint8_t> used_roles(role_count, 0);
  std::array<uint8_t, kIrInputKindCount> kind_ordinals = {};
  size_t lists_placed = 0;
  size_t scalar_done = 0;
  size_t scalar_idx = 0;
  for (size_t i = 0; i < work.node->inputs.size(); ++i) {
    const auto &in_node = work.node->inputs[i];
    if (in_node == nullptr || in_node->output == nullptr) {
      continue;
    }
    if (in_node->output->IsTensor()) {
      if (!PushResolvedTensorRebased(in_node, ctx, work, err)) {
        return false;
      }
    } else if (in_node->output->IsTuple()) {
      if (!PlaceOneShapeList(ctx, work, in_node, lists_placed, err)) {
        return false;
      }
    } else if (!ScanScalarOrAttrInput(in_node, ctx, work, scalars, used_roles, kind_ordinals, err)) {
      return false;
    }
    if (!FlushPendingScalars(ctx, work, scalars, scalar_done, scalar_idx, err)) {
      return false;
    }
  }
  if (!AppendDim0Axis(ctx, work, err)) {
    return false;
  }
  return CheckCollectedArity(work, err);
}

bool CollectAiCoreAsStridedInputs(LoweringContext &ctx, AiCoreOpWork &work, std::string *err) {
  std::vector<FoldedScalar> scalars;
  std::vector<fxrt::ir::NodePtr> packed_tuples;
  const OpBehavior *behavior = work.entry.behavior;
  const uint8_t role_count = (behavior != nullptr) ? behavior->input_role_count : 0;
  std::vector<uint8_t> used_roles(role_count, 0);
  std::array<uint8_t, kIrInputKindCount> kind_ordinals = {};
  for (size_t i = 0; i < work.node->inputs.size(); ++i) {
    const auto &in_node = work.node->inputs[i];
    if (in_node == nullptr || in_node->output == nullptr) {
      continue;
    }
    if (in_node->output->IsTensor()) {
      if (!PushResolvedTensor(in_node, ctx, work, err)) {
        return false;
      }
    } else if (in_node->output->IsTuple()) {
      packed_tuples.push_back(in_node);
    } else if (!ScanScalarOrAttrInput(in_node, ctx, work, scalars, used_roles, kind_ordinals, err)) {
      return false;
    }
  }
  if (!TuplesFillMissingTensorSlots(work, packed_tuples, scalars)) {
    return CollectAsStridedPathCInputs(ctx, work, packed_tuples, scalars, err);
  }
  size_t lists_placed = 0;
  if (std::any_of(packed_tuples.begin(), packed_tuples.end(), [&](const fxrt::ir::NodePtr &list_node) {
        return !PlaceOneShapeList(ctx, work, list_node, lists_placed, err);
      })) {
    return false;
  }
  size_t scalar_done = 0;
  size_t scalar_idx = 0;
  return FlushPendingScalars(ctx, work, scalars, scalar_done, scalar_idx, err) && CheckCollectedArity(work, err);
}

bool CollectAiCoreInputs(LoweringContext &ctx, AiCoreOpWork &work, std::string *err) {
  work.optional = DescribeAiCoreOptionalInputs(work.entry);
  if (!ValidateAiCoreInputRoles(work, err)) {
    return false;
  }
  if (work.entry.behavior != nullptr && work.entry.behavior->recipe == InputRecipe::kAsStrided) {
    return CollectAiCoreAsStridedInputs(ctx, work, err);
  }
  return CollectAiCoreTensorIoInputs(ctx, work, err);
}

}  // namespace fx_converter
