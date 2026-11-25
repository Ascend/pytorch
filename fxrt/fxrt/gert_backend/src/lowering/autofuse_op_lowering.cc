/**
 * FX Converter lowerings for allocation-only `aten.empty_strided` and the
 * backend-neutral compiled-kernel mutation HOP.
 */

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/graph_builder_utils.h"
#include "exe_graph/graph_skeleton_builder.h"
#include "lowering/common/wiring.h"
#include "lowering/op_lowering.h"
#include "lowering/op_lowering_registry.h"
#include "mrt_ge/mrt_node_utils.h"
#include "shape/fx_expr_infer_shape.h"
#include "shape/fx_sym_expr_parser.h"
#include "shape/mrt_to_ge_expr.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "ir/symbolic/symbolic.h"
#include "ops/op_def/ops_name.h"

namespace fx_converter {
namespace {

struct TensorArg {
  size_t kernel_arg_index = 0;
  size_t tensor_index = 0;
  TensorRef ref;
  fx_converter::Node *shape = nullptr;
  int32_t shape_out = 0;
  uint64_t shape_symbol = 0;
  fx_converter::Node *addr = nullptr;
  int32_t addr_out = 0;
  uint64_t addr_symbol = 0;
};

constexpr size_t kCompiledKernelRealArgOffset = 2U;

bool ReadStringAttr(const fxrt::ir::NodePtr &node, const char *name, std::string *value) {
  if (node == nullptr || value == nullptr || name == nullptr) {
    return false;
  }
  const auto it = node->attrs.find(name);
  if (it == node->attrs.end() || it->second == nullptr || !it->second->IsString()) {
    return false;
  }
  *value = it->second->ToString();
  return !value->empty();
}

std::shared_ptr<FxShapeRule> BuildRule(const fxrt::ir::NodePtr &node, LoweringContext &ctx, const char *source,
                                       std::string *err) {
  auto rule = BuildOutputSymbolicShapeRule(node, ctx.symbol_env, source);
  if (rule == nullptr) {
    if (err != nullptr) {
      *err = std::string("no symbolic output shape / symbol environment for ") + source;
    }
    return nullptr;
  }
  return rule;
}

fx_converter::Node *AddFxInfer(const fxrt::ir::NodePtr &node, const std::string &tag,
                         const std::vector<TensorRef> &inputs, LoweringContext &ctx,
                         const std::shared_ptr<FxShapeRule> &rule, uint64_t *shape_symbol) {
  const auto out_shape = ShapeOfNode(node);
  const auto out_dtype = DtypeOfNode(node);
  const auto td_out = MakeNdTensorDesc(out_shape, out_dtype);
  const ge::TensorDesc empty_td;
  const FxShapeRule *rule_ptr = rule.get();
  ctx.built->shape_rules.push_back(rule);
  auto *c_rule = AddConstNode(ctx.main, "const_shape_rule_" + tag, &rule_ptr, sizeof(rule_ptr), false, *ctx.sym_gen,
                              FxrtConstType::kVoidPtr);
  auto infer_func = GetFxExprInferShapeFunc();
  auto *c_infer = AddConstNode(ctx.main, "const_infer_func_" + tag, &infer_func, sizeof(infer_func), false,
                               *ctx.sym_gen, FxrtConstType::kKernelFuncPtr);
  ctx.kernel_nodes->insert(ctx.kernel_nodes->end(), {c_rule, c_infer});
  auto *infer = AddInferShapeNode(ctx.main, tag, inputs, td_out, empty_td, c_rule, c_infer, *ctx.sym_gen,
                                  shape_symbol);
  if (ctx.bind_symbols != nullptr) {
    fx_converter::AddDataEdge(ctx.main, ctx.bind_symbols, -1, infer, -1);
  }
  ctx.kernel_nodes->push_back(infer);
  return infer;
}

bool RebaseAutofuseInputOffset(const TensorRef &ref, const std::string &tag, size_t index,
                               LoweringContext &ctx, TensorArg *out) {
  const bool has_static_offset = ref.storage_offset_expr != nullptr || ref.storage_offset != 0;
  if (!ref.runtime_storage_offset.Valid() && !has_static_offset) return true;
  const int32_t item_size = ge::GetSizeByDataType(ref.dtype);
  if (item_size <= 0 ||
      (ref.storage_offset_expr == nullptr &&
       (ref.storage_offset < 0 ||
        static_cast<uint64_t>(ref.storage_offset) > std::numeric_limits<size_t>::max() /
                                                       static_cast<size_t>(item_size)))) {
    return false;
  }
  // Compiled kernels encode graph-local indexing in their generated code, but
  // their pointer argument must start at the effective input view origin. The
  // feed offset is refreshed per Execute; the view offset is static or carried
  // by its own runtime layout payload. Keep both separate from the TensorRef so
  // aliases and last-reader ownership retain the original storage coordinates.
  const ge::TensorDesc empty_td;
  RtGraphBuilder builder(ctx.main, empty_td, *ctx.sym_gen, ctx.kernel_nodes);
  const std::string suffix = tag + "_" + std::to_string(index);
  const int64_t zero_offset = 0;
  auto runtime_offset = ref.runtime_storage_offset;
  if (!runtime_offset.Valid()) {
    runtime_offset = builder.Const("const_autofuse_runtime_offset_" + suffix, &zero_offset,
                                  sizeof(zero_offset), false);
  }
  if (!runtime_offset.Valid()) return false;
  const size_t zero_bytes = 0U;
  const size_t view_offset_bytes = ref.storage_offset_expr == nullptr
                                       ? static_cast<size_t>(ref.storage_offset) * static_cast<size_t>(item_size)
                                       : zero_bytes;
  const size_t element_size = static_cast<size_t>(item_size);
  const auto byte_offset = builder.Const("const_autofuse_offset_" + suffix, &view_offset_bytes,
                                         sizeof(view_offset_bytes), false);
  const auto item = builder.Const("const_autofuse_itemsize_" + suffix, &element_size, sizeof(element_size), false);
  const TensorEndpoint address{out->addr, out->addr_out, out->addr_symbol};
  const TensorEndpoint shape{out->shape, out->shape_out, out->shape_symbol};
  std::vector<TensorEndpoint> inputs{address, shape, runtime_offset, byte_offset, item};
  if (ref.storage_offset_expr != nullptr) {
    TensorEndpoint view_offset;
    if (!AddRuntimeLayoutTensor(ctx.main, ctx.built, "autofuse_view_offset_" + suffix, {0},
                                {ref.storage_offset_expr}, *ctx.sym_gen, ctx.kernel_nodes, &view_offset)) {
      return false;
    }
    inputs.push_back(view_offset);
  }
  const auto view = builder.Node("FxContiguousView", "autofuse_input_origin_" + suffix, inputs, 2U);
  if (view.outs.size() != 2U || !view.outs[0].Valid()) return false;
  out->addr = view.outs[0].node;
  out->addr_out = view.outs[0].out;
  out->addr_symbol = view.outs[0].sym;
  return true;
}

bool WireAutofuseTensorArg(const TensorRef &ref, const std::string &tag, size_t index, LoweringContext &ctx,
                           TensorArg *out) {
  out->ref = ref;
  // Same Launch contract as Path C: (addr=GertTensorData*, shape=StorageShape*).
  // compiled_kernel tiling/dyn dims need ATen logical size, not physical storage.
  if (ref.logical_shape_ep.Valid()) {
    out->shape = ref.logical_shape_ep.node;
    out->shape_out = ref.logical_shape_ep.out;
    out->shape_symbol = ref.logical_shape_ep.sym;
  } else if (ref.HasUnpackedPair()) {
    out->shape = ref.unpacked.shape.node;
    out->shape_out = ref.unpacked.shape.out;
    out->shape_symbol = ref.unpacked.shape.sym;
  }
  if (ref.HasUnpackedPair()) {
    out->addr = ref.unpacked.addr.node;
    out->addr_out = ref.unpacked.addr.out;
    out->addr_symbol = ref.unpacked.addr.sym;
    return out->shape != nullptr && out->addr != nullptr && RebaseAutofuseInputOffset(ref, tag, index, ctx, out);
  }
  uint64_t shape_symbol = 0;
  uint64_t addr_symbol = 0;
  fx_converter::Node *split = CreateSplitDataTensor(ctx, ref, "autofuse_" + tag, index, &shape_symbol, &addr_symbol);
  if (split == nullptr) {
    return false;
  }
  if (out->shape == nullptr) {
    out->shape = split;
    out->shape_out = 0;
    out->shape_symbol = shape_symbol;
  }
  out->addr = split;
  out->addr_out = 1;
  out->addr_symbol = addr_symbol;
  return out->shape != nullptr && out->addr != nullptr && RebaseAutofuseInputOffset(ref, tag, index, ctx, out);
}

bool ReportError(std::string *err, const std::string &message) {
  if (err != nullptr) {
    *err = message;
  }
  return false;
}

struct CreatedAutofuseResource {
  fx_converter::Node *adapter_const = nullptr;
  fx_converter::Node *kernel_const = nullptr;
  fx_converter::Node *init = nullptr;
  uint64_t symbol = 0U;
};

CreatedAutofuseResource CreateAutofuseResource(LoweringContext &ctx, uint32_t slot, const std::string &adapter_path,
                                                const std::string &kernel_path) {
  CreatedAutofuseResource created;
  const ge::TensorDesc empty_td;
  const auto suffix = std::to_string(slot);
  created.adapter_const = AddConstNode(ctx.built->init.get(), "const_autofuse_adapter_" + suffix,
                                       adapter_path.c_str(), adapter_path.size() + 1U, true, *ctx.sym_gen);
  created.kernel_const = AddConstNode(ctx.built->init.get(), "const_autofuse_kernel_" + suffix,
                                      kernel_path.c_str(), kernel_path.size() + 1U, true, *ctx.sym_gen);
  if (created.adapter_const == nullptr || created.kernel_const == nullptr) {
    return created;
  }
  auto init_desc = std::make_shared<fx_converter::FxNodeSpec>("autofuse_init_" + suffix, "InductorAutofuseInit");
  init_desc->AddInputDesc(empty_td);
  init_desc->AddInputDesc(empty_td);
  init_desc->AddOutputDesc(empty_td);
  created.init = fx_converter::AddNode(ctx.built->init.get(), init_desc);
  if (created.init == nullptr || created.adapter_const->GetExtendInfo() == nullptr ||
      created.kernel_const->GetExtendInfo() == nullptr) {
    return created;
  }
  created.symbol = (*ctx.sym_gen)++;
  WireSymbols(created.init,
              {created.adapter_const->GetExtendInfo()->GetOutputSymbol(0),
               created.kernel_const->GetExtendInfo()->GetOutputSymbol(0)},
              {created.symbol});
  fx_converter::AddDataEdge(ctx.built->init.get(), created.adapter_const, 0, created.init, 0);
  fx_converter::AddDataEdge(ctx.built->init.get(), created.kernel_const, 0, created.init, 1);
  return created;
}

bool EnsureAutofuseResource(LoweringContext &ctx, const std::string &adapter_path, const std::string &kernel_path,
                            uint64_t *resource_symbol, std::string *err) {
  if (ctx.autofuse_resources == nullptr || resource_symbol == nullptr) {
    return ReportError(err, "missing Autofuse resource cache");
  }
  const std::string key = adapter_path + "\n" + kernel_path;
  const auto found = ctx.autofuse_resources->find(key);
  if (found != ctx.autofuse_resources->end()) {
    *resource_symbol = found->second;
    return true;
  }
  // Use the number of resources already cached in this graph as a stable unique
  // suffix; otherwise every distinct autofuse kernel would try to create the
  // same const_autofuse_*_0 and autofuse_init_0 nodes.
  const auto slot = static_cast<uint32_t>(ctx.autofuse_resources->size());
  const auto created = CreateAutofuseResource(ctx, slot, adapter_path, kernel_path);
  if (created.adapter_const == nullptr || created.kernel_const == nullptr || created.init == nullptr) {
    return ReportError(err, "failed to create unique Autofuse resource nodes");
  }
  // The Init resource is already visible to later nodes by its endpoint symbol.
  if (ctx.autofuse_resource_nodes != nullptr) {
    (*ctx.autofuse_resource_nodes)[key] = created.init;
  }
  ctx.kernel_nodes->insert(ctx.kernel_nodes->end(), {created.adapter_const, created.kernel_const, created.init});
  (*ctx.autofuse_resources)[key] = created.symbol;
  *resource_symbol = created.symbol;
  return true;
}

bool ReadMutationIndices(const std::vector<fxrt::ir::NodePtr> &args, std::vector<size_t> *mutated,
                         std::string *err) {
  const auto mutation_node = args[1];
  if (mutation_node == nullptr || mutation_node->output == nullptr || !mutation_node->output->IsTuple()) {
    return ReportError(err, "compiled_kernel_mutation mutated_arg_indices is not a tuple");
  }
  for (const auto &index : *mutation_node->output->ToTuple()) {
    if (index == nullptr || !index->IsInt() || index->ToInt() < 0) {
      return ReportError(err, "compiled_kernel_mutation has invalid mutated arg index");
    }
    const auto mutated_index = static_cast<size_t>(index->ToInt());
    if (std::find(mutated->begin(), mutated->end(), mutated_index) != mutated->end()) {
      return ReportError(err, "compiled_kernel_mutation has duplicate mutated arg indices");
    }
    mutated->push_back(mutated_index);
  }
  if (mutated->empty()) {
    return ReportError(err, "compiled_kernel_mutation has no mutated args");
  }
  return true;
}

bool CollectTensorArguments(const std::vector<fxrt::ir::NodePtr> &args, const std::string &tag,
                            LoweringContext &ctx, std::vector<TensorArg> *tensors, std::string *err) {
  for (size_t raw_index = kCompiledKernelRealArgOffset; raw_index < args.size(); ++raw_index) {
    const auto &arg = args[raw_index];
    if (arg == nullptr || arg->output == nullptr || !arg->output->IsTensor()) {
      continue;
    }
    const auto name_it = ctx.name_of->find(arg.get());
    if (name_it == ctx.name_of->end()) {
      return ReportError(err, "compiled_kernel_mutation tensor argument has no producer name");
    }
    const auto ref_it = ctx.tensors->find(name_it->second);
    if (ref_it == ctx.tensors->end()) {
      return ReportError(err, "compiled_kernel_mutation tensor argument is not lowered");
    }
    TensorArg tensor;
    tensor.kernel_arg_index = raw_index - kCompiledKernelRealArgOffset;
    tensor.tensor_index = tensors->size();
    if (!WireAutofuseTensorArg(ref_it->second, tag, tensor.tensor_index, ctx, &tensor)) {
      return ReportError(err, "compiled_kernel_mutation failed to wire tensor argument");
    }
    tensors->push_back(std::move(tensor));
  }
  if (tensors->empty()) {
    return ReportError(err, "compiled_kernel_mutation has no tensor arguments");
  }
  return true;
}

// Hop args after the tensors are size SymInts / ints. Adapter tiling maps use
// ir_input_index = tensor_slot_num + k to read the k-th of those scalars — not
// flattened StorageShape dims of the tensor args (that yields B,H,1 instead of B,H,S).
void CollectScalarArguments(const std::vector<fxrt::ir::NodePtr> &args, std::vector<fxrt::ir::NodePtr> *scalars) {
  for (size_t raw_index = kCompiledKernelRealArgOffset; raw_index < args.size(); ++raw_index) {
    const auto &arg = args[raw_index];
    if (arg == nullptr || arg->output == nullptr) {
      continue;
    }
    if (arg->output->IsInt() || arg->output->IsSymbol()) {
      scalars->push_back(arg);
    }
  }
}

std::shared_ptr<FxShapeRule> BuildDynShapeRule(const std::vector<fxrt::ir::NodePtr> &scalars, FxSymbolEnv *env,
                                               std::string *err) {
  if (env == nullptr) {
    if (err != nullptr) {
      *err = "compiled_kernel_mutation dyn-shape rule needs a SymbolEnv";
    }
    return nullptr;
  }
  auto rule = std::make_shared<FxShapeRule>();
  rule->env = env;
  rule->source = "autofuse_dyn_shapes";
  for (const auto &arg : scalars) {
    fxrt::ir::SymbolicExprPtr expr;
    if (arg->output->IsInt()) {
      expr = fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicConst>(arg->output->ToInt());
    } else {
      // ABI bridge flattens dims to one SymbolicVar; ParseFxSymExpr rebuilds the tree and
      // interns free vars through EnsureIrVar so BindSymbolEnv::SetValue reaches Evaluate().
      expr = ParseFxSymExpr(arg->output->ToSymbol(), *env);
    }
    if (expr == nullptr) {
      if (err != nullptr) {
        *err = "compiled_kernel_mutation scalar arg has no symbolic expression";
      }
      return nullptr;
    }
    rule->out_dims_ir.push_back(expr);
  }
  return rule;
}

fx_converter::Node *AddDynShapeInfer(const std::string &tag, const std::vector<fxrt::ir::NodePtr> &scalars,
                               LoweringContext &ctx, const std::shared_ptr<FxShapeRule> &rule,
                               uint64_t *shape_symbol) {
  // GE BuildInferShapeOutputs needs a non-null output TensorDesc. A rank-N desc whose
  // dims are all -1 can leave GetOutputDesc null on this CANN build; use positive
  // placeholders (overwritten by FxExprInferFromRule at Execute).
  std::vector<int64_t> hint(scalars.size(), 1);
  for (size_t i = 0; i < scalars.size(); ++i) {
    if (scalars[i]->output != nullptr && scalars[i]->output->IsInt()) {
      const int64_t v = scalars[i]->output->ToInt();
      if (v > 0) {
        hint[i] = v;
      }
    }
  }
  const auto td_out = MakeNdTensorDesc(hint, ge::DT_INT64);
  const ge::TensorDesc empty_td;
  const FxShapeRule *rule_ptr = rule.get();
  ctx.built->shape_rules.push_back(rule);
  auto *c_rule =
      AddConstNode(ctx.main, "const_dyn_shape_rule_" + tag, &rule_ptr, sizeof(rule_ptr), false, *ctx.sym_gen,
                   FxrtConstType::kVoidPtr);
  auto infer_func = GetFxExprInferShapeFunc();
  auto *c_infer = AddConstNode(ctx.main, "const_dyn_infer_func_" + tag, &infer_func, sizeof(infer_func), false,
                               *ctx.sym_gen, FxrtConstType::kKernelFuncPtr);
  ctx.kernel_nodes->insert(ctx.kernel_nodes->end(), {c_rule, c_infer});
  // No tensor inputs: dims come only from the SymbolEnv-backed rule (empty_strided pattern).
  auto *infer =
      AddInferShapeNode(ctx.main, tag + "_dyn", {}, td_out, empty_td, c_rule, c_infer, *ctx.sym_gen, shape_symbol);
  if (ctx.bind_symbols != nullptr) {
    fx_converter::AddDataEdge(ctx.main, ctx.bind_symbols, -1, infer, -1);
  }
  ctx.kernel_nodes->push_back(infer);
  return infer;
}

bool ValidateMutationArguments(const std::vector<fxrt::ir::NodePtr> &args, const std::vector<size_t> &mutated,
                               std::string *err) {
  const size_t kernel_arg_num = args.size() - kCompiledKernelRealArgOffset;
  for (const auto index : mutated) {
    const size_t raw_index = kCompiledKernelRealArgOffset + index;
    if (index >= kernel_arg_num || args[raw_index] == nullptr || args[raw_index]->output == nullptr ||
        !args[raw_index]->output->IsTensor()) {
      return ReportError(err, "compiled_kernel_mutation mutated index is not a tensor argument");
    }
  }
  return true;
}

bool GetMainResourceNode(LoweringContext &ctx, const std::string &key, fx_converter::Node **resource_node,
                         std::string *err) {
  if (ctx.autofuse_resource_nodes == nullptr) {
    return ReportError(err, "missing Autofuse resource node cache");
  }
  const auto resource_it = ctx.autofuse_resource_nodes->find(key);
  if (resource_it == ctx.autofuse_resource_nodes->end() || resource_it->second == nullptr) {
    return ReportError(err, "Autofuse resource is not available in the Main execute graph");
  }
  *resource_node = resource_it->second;
  return true;
}

struct AutofuseConstants {
  fx_converter::Node *slot_num = nullptr;
  fx_converter::Node *input_num = nullptr;
  fx_converter::Node *output_num = nullptr;
  uint32_t tensor_slot_count = 0U;
};

AutofuseConstants AddAutofuseConstants(const std::string &tag, const std::vector<TensorArg> &tensors,
                                       const std::vector<size_t> &mutated, LoweringContext &ctx) {
  AutofuseConstants constants;
  constants.tensor_slot_count = static_cast<uint32_t>(tensors.size());
  const uint32_t output_count = static_cast<uint32_t>(mutated.size());
  const uint32_t input_count = constants.tensor_slot_count - output_count;
  constants.slot_num = AddConstNode(ctx.main, "const_autofuse_tensor_slots_" + tag,
                                    &constants.tensor_slot_count, sizeof(constants.tensor_slot_count), false,
                                    *ctx.sym_gen, FxrtConstType::kUInt32T);
  constants.input_num = AddConstNode(ctx.main, "const_autofuse_input_num_" + tag, &input_count,
                                     sizeof(input_count), false, *ctx.sym_gen, FxrtConstType::kUInt32T);
  constants.output_num = AddConstNode(ctx.main, "const_autofuse_output_num_" + tag, &output_count,
                                      sizeof(output_count), false, *ctx.sym_gen, FxrtConstType::kUInt32T);
  ctx.kernel_nodes->insert(ctx.kernel_nodes->end(),
                           {constants.slot_num, constants.input_num, constants.output_num});
  return constants;
}

struct AutofusePrepareNodes {
  fx_converter::Node *tiling = nullptr;
  fx_converter::Node *workspace = nullptr;
  uint64_t workspace_size_symbol = 0U;
  uint64_t block_symbol = 0U;
  uint64_t tiling_symbol = 0U;
  uint64_t workspace_addr_symbol = 0U;
};

AutofusePrepareNodes AddAutofusePrepare(const std::string &tag, const std::vector<TensorArg> &tensors,
                                        const AutofuseConstants &constants, fx_converter::Node *resource_node,
                                        uint64_t resource_symbol, fx_converter::Node *dyn_shape_node,
                                        uint64_t dyn_shape_symbol, int32_t dyn_shape_out, LoweringContext &ctx) {
  AutofusePrepareNodes nodes;
  const ge::TensorDesc empty_td;
  auto tiling_desc = std::make_shared<fx_converter::FxNodeSpec>("autofuse_tiling_" + tag, "InductorAutofuseTiling");
  tiling_desc->AddInputDesc(empty_td);
  tiling_desc->AddInputDesc(empty_td);
  // Dyn dims for adapter tiling maps: one StorageShape whose dims are the hop
  // scalar args (SymInt sizes), NOT the flattened tensor StorageShapes.
  if (dyn_shape_node != nullptr) {
    tiling_desc->AddInputDesc(empty_td);
  }
  tiling_desc->AddOutputDesc(empty_td);
  tiling_desc->AddOutputDesc(empty_td);
  tiling_desc->AddOutputDesc(empty_td);
  nodes.tiling = fx_converter::AddNode(ctx.main, tiling_desc);
  nodes.workspace_size_symbol = (*ctx.sym_gen)++;
  nodes.block_symbol = (*ctx.sym_gen)++;
  nodes.tiling_symbol = (*ctx.sym_gen)++;
  std::vector<uint64_t> tiling_symbols{resource_symbol, constants.slot_num->GetExtendInfo()->GetOutputSymbol(0)};
  if (dyn_shape_node != nullptr) {
    tiling_symbols.push_back(dyn_shape_symbol);
  }
  WireSymbols(nodes.tiling, tiling_symbols,
              {nodes.workspace_size_symbol, nodes.block_symbol, nodes.tiling_symbol});
  fx_converter::AddDataEdge(ctx.main, resource_node, 0, nodes.tiling, 0);
  fx_converter::AddDataEdge(ctx.main, constants.slot_num, 0, nodes.tiling, 1);
  if (dyn_shape_node != nullptr) {
    fx_converter::AddDataEdge(ctx.main, dyn_shape_node, dyn_shape_out, nodes.tiling, 2);
  }
  (void)tensors;

  auto workspace_desc = std::make_shared<fx_converter::FxNodeSpec>("autofuse_workspace_" + tag, "AllocBatchHbm");
  workspace_desc->AddInputDesc(empty_td);
  workspace_desc->AddInputDesc(empty_td);
  workspace_desc->AddOutputDesc(empty_td);
  nodes.workspace = fx_converter::AddNode(ctx.main, workspace_desc);
  nodes.workspace_addr_symbol = (*ctx.sym_gen)++;
  WireSymbols(nodes.workspace, {ctx.rt->s_allocator, nodes.workspace_size_symbol},
              {nodes.workspace_addr_symbol});
  fx_converter::AddDataEdge(ctx.main, ctx.rt->select_l2, 0, nodes.workspace, 0);
  fx_converter::AddDataEdge(ctx.main, nodes.tiling, 0, nodes.workspace, 1);
  return nodes;
}

fx_converter::Node *AddAutofuseLaunch(const std::string &tag, const std::vector<TensorArg> &tensors,
                                const AutofuseConstants &constants, const AutofusePrepareNodes &prepare,
                                fx_converter::Node *resource_node, uint64_t resource_symbol, fx_converter::Node *dyn_shape_node,
                                uint64_t dyn_shape_symbol, int32_t dyn_shape_out, LoweringContext &ctx) {
  constexpr size_t kFixedInputCount = 7U;
  const ge::TensorDesc empty_td;
  const size_t shape_slots = dyn_shape_node != nullptr ? 1U : tensors.size();
  auto launch_desc = std::make_shared<fx_converter::FxNodeSpec>("autofuse_launch_" + tag, "InductorAutofuseLaunch");
  // OM2 collection needs to distinguish the normal [tensor_0..N, shape_0..N]
  // layout from the dynamic-scalar layout [tensor_0..N, shared_shape]. Keep
  // this distinction explicit instead of inferring it from an odd input count.
  (void)fx_converter::FxAttrUtils::SetBool(launch_desc, "fx_shared_shape", dyn_shape_node != nullptr);
  for (size_t i = 0; i < kFixedInputCount + tensors.size() + shape_slots; ++i) {
    launch_desc->AddInputDesc(empty_td);
  }
  auto *launch = fx_converter::AddNode(ctx.main, launch_desc);
  std::vector<uint64_t> launch_symbols{
      resource_symbol, prepare.block_symbol, ctx.rt->s_stream, prepare.tiling_symbol,
      prepare.workspace_addr_symbol, constants.input_num->GetExtendInfo()->GetOutputSymbol(0),
      constants.output_num->GetExtendInfo()->GetOutputSymbol(0)};
  // The kernel reads and writes the same argument. A fresh model-output allocation
  // is still zero here, so pointing a returned relu/sigmoid at it yields the
  // activation of zeros. NetOutput copies the producer buffer after this launch.
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(launch_symbols),
                 [](const TensorArg &tensor) { return tensor.addr_symbol; });
  if (dyn_shape_node != nullptr) {
    launch_symbols.push_back(dyn_shape_symbol);
  } else {
    std::transform(tensors.begin(), tensors.end(), std::back_inserter(launch_symbols),
                   [](const TensorArg &tensor) { return tensor.shape_symbol; });
  }
  WireSymbols(launch, launch_symbols, {});
  for (size_t i = 0; i < tensors.size(); ++i) {
    fx_converter::AddDataEdge(ctx.main, tensors[i].addr, tensors[i].addr_out, launch,
                              static_cast<int32_t>(i + kFixedInputCount));
  }
  if (dyn_shape_node != nullptr) {
    fx_converter::AddDataEdge(ctx.main, dyn_shape_node, dyn_shape_out, launch,
                              static_cast<int32_t>(kFixedInputCount + tensors.size()));
  } else {
    for (size_t i = 0; i < tensors.size(); ++i) {
      fx_converter::AddDataEdge(ctx.main, tensors[i].shape, tensors[i].shape_out, launch,
                        static_cast<int32_t>(i + kFixedInputCount + tensors.size()));
    }
  }
  fx_converter::AddDataEdge(ctx.main, resource_node, 0, launch, 0);
  fx_converter::AddDataEdge(ctx.main, prepare.tiling, 1, launch, 1);
  fx_converter::AddDataEdge(ctx.main, ctx.rt->split_streams, 0, launch, 2);
  fx_converter::AddDataEdge(ctx.main, prepare.tiling, 2, launch, 3);
  fx_converter::AddDataEdge(ctx.main, prepare.workspace, 0, launch, 4);
  fx_converter::AddDataEdge(ctx.main, constants.input_num, 0, launch, 5);
  fx_converter::AddDataEdge(ctx.main, constants.output_num, 0, launch, 6);

  auto free_desc = std::make_shared<fx_converter::FxNodeSpec>("autofuse_free_workspace_" + tag, "FreeBatchHbm");
  free_desc->AddInputDesc(empty_td);
  (void)fx_converter::FxAttrUtils::SetInt(free_desc, "ReleaseResourceIndex", 0);
  auto *free_workspace = fx_converter::AddNode(ctx.main, free_desc);
  WireSymbols(free_workspace, {prepare.workspace_addr_symbol}, {});
  fx_converter::AddDataEdge(ctx.main, prepare.workspace, 0, free_workspace, 0);
  fx_converter::AddDataEdge(ctx.main, launch, -1, free_workspace, -1);
  ctx.kernel_nodes->insert(ctx.kernel_nodes->end(),
                           {prepare.tiling, prepare.workspace, launch, free_workspace});
  return launch;
}

// One TensorRef per mutated argument, in tuple-slot order. tuple_getitem aliases
// the recorded slot; publishing only mutated.front() leaves every later getitem
// without a producer.
bool PublishInPlaceSlots(const std::vector<TensorArg> &tensors, const std::vector<size_t> &mutated,
                         LoweringResult *out, std::string *err) {
  out->outputs.clear();
  out->outputs.reserve(mutated.size());
  for (const size_t index : mutated) {
    const auto found = std::find_if(tensors.begin(), tensors.end(), [index](const TensorArg &tensor) {
      return tensor.kernel_arg_index == index;
    });
    if (found == tensors.end()) {
      return ReportError(err, "compiled_kernel_mutation output tensor is not available");
    }
    out->outputs.push_back(found->ref);
  }
  return true;
}

bool SetAutofuseOutput(const fxrt::ir::NodePtr &node, const std::vector<TensorArg> &tensors,
                       const std::vector<size_t> &mutated, LoweringResult *out, std::string *err) {
  if (node->output != nullptr && node->output->IsTuple()) {
    return PublishInPlaceSlots(tensors, mutated, out, err);
  }
  // Publish the buffer the kernel wrote. A separate OutputData is still empty
  // when an in-place relu/sigmoid runs, and returning it drops the producer.
  out->outputs.resize(1);
  const auto mutated_ref = std::find_if(tensors.begin(), tensors.end(), [&](const TensorArg &tensor) {
    return tensor.kernel_arg_index == mutated.front();
  });
  if (mutated_ref == tensors.end()) {
    return ReportError(err, "compiled_kernel_mutation output tensor is not available");
  }
  out->outputs.front() = mutated_ref->ref;
  return true;
}

// Tuple outputs take their shape from the first mutated tensor. A count mismatch
// would leave tuple_getitem slots unaligned with the kernel writes.
bool ResolveAutofuseShapeNode(const fxrt::ir::NodePtr &node, const std::vector<fxrt::ir::NodePtr> &args,
                              const std::vector<size_t> &mutated, fxrt::ir::NodePtr *shape_node, std::string *err) {
  *shape_node = node;
  if (node->output == nullptr || !node->output->IsTuple()) {
    return true;
  }
  const auto output_tuple = node->output->ToTuple();
  if (output_tuple == nullptr || output_tuple->Size() != mutated.size()) {
    return ReportError(err, "compiled_kernel_mutation tuple output count does not match mutated arguments");
  }
  const size_t shape_arg = kCompiledKernelRealArgOffset + mutated.front();
  if (shape_arg >= args.size() || args[shape_arg] == nullptr || args[shape_arg]->output == nullptr ||
      !args[shape_arg]->output->IsTensor()) {
    return ReportError(err, "compiled_kernel_mutation tuple output has no tensor shape anchor");
  }
  *shape_node = args[shape_arg];
  return true;
}

void SetAutofuseCni(const fxrt::ir::NodePtr &node, const std::vector<TensorArg> &tensors,
                    fx_converter::Node *infer, fx_converter::Node *tiling, fx_converter::Node *launch, LoweringResult *out) {
  out->launch = launch;
  out->prepare = tiling;
  out->cni.op_type = "InductorAutofuse";
  out->cni.input_shapes.reserve(tensors.size());
  out->cni.input_dtypes.reserve(tensors.size());
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(out->cni.input_shapes),
                 [](const TensorArg &tensor) {
                   return tensor.ref.logical_shape.empty() ? tensor.ref.shape : tensor.ref.logical_shape;
                 });
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(out->cni.input_dtypes),
                 [](const TensorArg &tensor) { return static_cast<int32_t>(tensor.ref.dtype); });
  out->cni.output_shape = ShapeOfNode(node);
  out->cni.output_dtype = static_cast<int32_t>(DtypeOfNode(node));
  out->cni.compute_nodes = {infer, tiling, launch};
}

class CompiledKernelMutationLowering final : public OpLowering {
 public:
  const char *Name() const override { return "compiled_kernel_mutation"; }
  bool MayWriteInputs() const override { return true; }
  InputLayoutKind InputLayout(const fxrt::ir::NodePtr &, size_t) const override {
    return InputLayoutKind::kStorageIndexed;
  }
  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *) const override {
    return node != nullptr && node->op == fxrt::ops::Op_compiled_kernel_mutation;
  }

  // autofuse_call reaches the IR under its own op; keep the FX-level name supported.
  std::vector<std::string> SupportedLogicalOps() const override {
    return {"compiled_kernel_mutation", "autofuse_call"};
  }

  bool Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &ctx, LoweringResult *out,
             std::string *err) override {
    if (out == nullptr || !ValidateLoweringContext(ctx, false, Name(), err)) {
      return false;
    }
    std::string adapter_path;
    std::string kernel_path;
    if (!ReadStringAttr(node, "adapter_path", &adapter_path) || !ReadStringAttr(node, "kernel_path", &kernel_path)) {
      return ReportError(err, "compiled_kernel_mutation requires backend adapter_path and kernel_path metadata");
    }
    const auto &args = node->inputs;
    if (args.size() <= kCompiledKernelRealArgOffset) {
      return ReportError(err, "compiled_kernel_mutation has no kernel arguments");
    }
    std::vector<size_t> mutated;
    if (!ReadMutationIndices(args, &mutated, err)) {
      return false;
    }
    std::vector<TensorArg> tensors;
    if (!CollectTensorArguments(args, tag, ctx, &tensors, err) || !ValidateMutationArguments(args, mutated, err)) {
      return false;
    }
    std::vector<fxrt::ir::NodePtr> scalars;
    CollectScalarArguments(args, &scalars);
    fx_converter::Node *dyn_shape_node = nullptr;
    uint64_t dyn_shape_symbol = 0;
    if (!scalars.empty()) {
      auto dyn_rule = BuildDynShapeRule(scalars, ctx.symbol_env, err);
      if (dyn_rule == nullptr) {
        return false;
      }
      dyn_shape_node = AddDynShapeInfer(tag, scalars, ctx, dyn_rule, &dyn_shape_symbol);
    }
    fxrt::ir::NodePtr shape_node;
    if (!ResolveAutofuseShapeNode(node, args, mutated, &shape_node, err)) {
      return false;
    }
    auto rule = BuildRule(shape_node, ctx, "compiled_kernel_mutation", err);
    if (rule == nullptr) {
      return false;
    }
    std::vector<TensorRef> tensor_refs;
    tensor_refs.reserve(tensors.size());
    std::transform(tensors.begin(), tensors.end(), std::back_inserter(tensor_refs),
                   [](const TensorArg &tensor) { return tensor.ref; });
    uint64_t shape_symbol = 0;
    auto *infer = AddFxInfer(shape_node, tag, tensor_refs, ctx, rule, &shape_symbol);
    (void)shape_symbol;

    uint64_t resource_symbol = 0;
    if (!EnsureAutofuseResource(ctx, adapter_path, kernel_path, &resource_symbol, err)) {
      return false;
    }
    fx_converter::Node *resource_node = nullptr;
    if (!GetMainResourceNode(ctx, adapter_path + "\n" + kernel_path, &resource_node, err)) {
      return false;
    }
    const auto constants = AddAutofuseConstants(tag, tensors, mutated, ctx);
    const auto prepare = AddAutofusePrepare(tag, tensors, constants, resource_node, resource_symbol, dyn_shape_node,
                                            dyn_shape_symbol, 0, ctx);

    // In-place on the producer buffer. NetOutput copies a returned buffer after
    // this launch; writing straight into a new model output reads zeros.
    auto *launch = AddAutofuseLaunch(tag, tensors, constants, prepare, resource_node, resource_symbol, dyn_shape_node,
                                     dyn_shape_symbol, 0, ctx);
    if (!SetAutofuseOutput(node, tensors, mutated, out, err)) {
      return false;
    }
    SetAutofuseCni(shape_node, tensors, infer, prepare.tiling, launch, out);
    return true;
  }
};

}  // namespace

}  // namespace fx_converter

namespace {
::fx_converter::OpLoweringRegistrar g_compiled_kernel_mutation_lowering(
    21, std::make_unique<::fx_converter::CompiledKernelMutationLowering>());
}  // namespace
