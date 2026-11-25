/**
 * Lowest-priority fallback for Op_custom_call.
 *
 * Torch runs in the ABI=1 bridge, waits c10d async collectives, then
 * aclrtMemcpyAsync-copies the contiguous result into GE memory.
 * Only versioned POD views cross the ABI boundary.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "exe_graph/execute_graph_builder.h"
#include "exe_graph/graph_builder_utils.h"
#include "ir/symbolic/symbolic.h"
#include "ir/value/value.h"
#include "lowering/common/view_utils.h"
#include "lowering/common/wiring.h"
#include "lowering/op_lowering.h"
#include "lowering/op_lowering_registry.h"
#include "lowering/torch_fallback/torch_fallback_runtime.h"
#include "mrt_ge/mrt_node_utils.h"
#include "ops/op_def/ops_name.h"
#include "shape/fx_expr_infer_shape.h"
#include "shape/mrt_to_ge_expr.h"
#include "shape/shape_list_utils.h"
#include "shape/fx_sym_expr_parser.h"

namespace fx_converter {
namespace {

constexpr int kTorchFallbackPriority = 10000;
// Matches EagerOpExecutionContext additional inputs plus the slots ExecuteCustomOp
// used after them: allocator, stream, call, attached stream, graph id, node id, core num.
constexpr size_t kCustomOpExtraInputs = 7U;

struct FallbackInputs {
  std::vector<TorchFallbackArgument> arguments;
  std::vector<TensorRef> tensors;
};

struct FallbackOutput {
  fx_converter::Node *infer = nullptr;
  // Out-schema InferShape. Launch wiring later stores SplitDataTensor in infer.
  fx_converter::Node *shape_rule_infer = nullptr;
  fx_converter::Node *output_data = nullptr;
  fx_converter::Node *alloc = nullptr;
  fx_converter::Node *source_alloc = nullptr;
  fx_converter::Node *destination_alloc = nullptr;
  bool deferred_shape = false;
  fx_converter::Node *tensor_attr = nullptr;
  uint64_t source_address_symbol = 0;
  uint64_t shape_symbol = 0;
  uint64_t output_data_symbol = 0;
  uint64_t address_symbol = 0;
};

struct FallbackOutputSpec {
  fxrt::ir::NodePtr shape_node;
  std::vector<int64_t> shape;
  ge::DataType dtype = ge::DT_FLOAT;
  int8_t python_dense = -1;
};

std::vector<int8_t> ReadPythonOutputDenseFlags(const fxrt::ir::NodePtr &producer, size_t output_count,
                                             const std::string &attr_name = "fxrt_fallback_output_dense") {
  if (producer == nullptr) {
    return {};
  }
  const auto attr = producer->attrs.find(attr_name);
  if (attr == producer->attrs.end() || attr->second == nullptr || !attr->second->IsTuple()) {
    return {};
  }
  const auto &flags = attr->second->ToTuple();
  if (flags == nullptr || flags->Size() != output_count) {
    return {};
  }
  std::vector<int8_t> result;
  result.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    const auto &flag = (*flags)[i];
    result.push_back(flag != nullptr && flag->IsBool() && flag->ToBool() ? 1 : 0);
  }
  return result;
}

std::vector<std::pair<size_t, size_t>> ReadFallbackReadAliases(const fxrt::ir::NodePtr &producer,
                                                             size_t output_count) {
  std::vector<std::pair<size_t, size_t>> pairs;
  const auto attribute = producer->attrs.find("fxrt_fallback_read_aliases");
  if (attribute == producer->attrs.end() || attribute->second == nullptr || !attribute->second->IsTuple()) {
    return pairs;
  }
  for (const auto &value : *attribute->second->ToTuple()) {
    if (value == nullptr || !value->IsTuple() || value->ToTuple()->Size() != 2U) continue;
    const auto &pair = value->ToTuple();
    if ((*pair)[0] == nullptr || (*pair)[1] == nullptr || !(*pair)[0]->IsInt() || !(*pair)[1]->IsInt()) continue;
    const int64_t input = (*pair)[0]->ToInt();
    const int64_t output = (*pair)[1]->ToInt();
    if (input >= 0 && output >= 0 && static_cast<size_t>(output) < output_count) {
      pairs.emplace_back(static_cast<size_t>(input), static_cast<size_t>(output));
    }
  }
  return pairs;
}

bool EnsureStaticSymbolicShape(const fxrt::ir::ValuePtr &value) {
  if (value == nullptr || !value->IsTensor()) {
    return false;
  }
  const auto tensor = value->ToTensor();
  if (tensor == nullptr || tensor->HasSymbolicShape()) {
    return tensor != nullptr;
  }
  const auto &shape = tensor->Shape();
  if (shape.empty() || std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim < 0; })) {
    return false;
  }
  std::vector<fxrt::ir::SymbolicExprPtr> symbolic_shape;
  symbolic_shape.reserve(shape.size());
  std::transform(shape.begin(), shape.end(), std::back_inserter(symbolic_shape), [](int64_t dim) {
    return fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicConst>(dim);
  });
  tensor->SetSymbolicShape(symbolic_shape);
  return true;
}

bool CollectFallbackOutputs(const fxrt::ir::NodePtr &producer, std::vector<FallbackOutputSpec> *outputs,
                            std::string *error) {
  if (producer == nullptr || producer->output == nullptr) {
    if (error != nullptr) {
      *error = "torch fallback: custom call has no output metadata";
    }
    return false;
  }
  std::vector<fxrt::ir::ValuePtr> values;
  if (producer->output->IsTensor()) {
    values.push_back(producer->output);
  } else if (producer->output->IsTuple()) {
    const auto &tuple = producer->output->ToTuple();
    if (tuple != nullptr) {
      values.assign(tuple->begin(), tuple->end());
    }
  }
  if (values.empty()) {
    if (error != nullptr) {
      *error = "torch fallback: output must be a Tensor or a fixed tuple/list of Tensors";
    }
    return false;
  }
  const auto python_dense = ReadPythonOutputDenseFlags(producer, values.size());
  outputs->reserve(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const auto &value = values[i];
    if (value == nullptr || !value->IsTensor()) {
      if (error != nullptr) {
        *error = "torch fallback: tuple/list output " + std::to_string(i) + " is not a Tensor";
      }
      return false;
    }
    (void)EnsureStaticSymbolicShape(value);
    auto shape_node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
    shape_node->op = fxrt::ops::Op_End;
    shape_node->output = value;
    FallbackOutputSpec spec;
    spec.shape_node = std::move(shape_node);
    spec.shape = ShapeOfNode(spec.shape_node);
    spec.dtype = DtypeOfNode(spec.shape_node);
    if (i < python_dense.size()) {
      spec.python_dense = python_dense[i];
    }
    outputs->push_back(std::move(spec));
  }
  return true;
}

bool ReadQualifiedName(const fxrt::ir::NodePtr &node, std::string *name, std::string *overload,
                       std::string *error) {
  if (node == nullptr || node->inputs.empty() || node->inputs[0] == nullptr ||
      node->inputs[0]->output == nullptr || !node->inputs[0]->output->IsString()) {
    if (error != nullptr) {
      *error = "torch fallback: custom call has no qualified operator name";
    }
    return false;
  }
  *name = node->inputs[0]->output->ToString();
  // Legacy graphs may still carry namespace.op.overload in the single name.
  const auto dot = name->rfind('.');
  // Export normally uses dispatcher::operator.overload, but custom namespaces
  // may arrive as namespace.operator.overload.  In either form, a second dot
  // proves that the final component is an overload rather than part of the
  // qualified operator name.
  if (overload != nullptr && overload->empty() && name->find('.') != dot &&
      dot != std::string::npos && dot + 1U < name->size()) {
    *overload = name->substr(dot + 1U);
    *name = name->substr(0U, dot);
  }
  return !name->empty();
}

bool HasUnseededShapeSymbol(const fxrt::ir::SymbolicExprPtr &expr, FxSymbolEnv &env) {
  if (expr == nullptr) {
    return false;
  }
  // ABI-0 exports flatten non-constant shape expressions into one SymbolicVar
  // named after the expression (for example "(6 * s72)"). Rebuild that tree
  // before checking seeds; treating the wrapper as a free variable rejects
  // valid multi-output fallback shapes.
  const auto parsed = ParseFxSymExpr(expr, env);
  const auto &candidate = parsed != nullptr ? parsed : expr;
  using Kind = fxrt::ir::SymbolicExpr::Kind;
  if (candidate->GetKind() == Kind::Variable) {
    const auto *var = static_cast<const fxrt::ir::SymbolicVar *>(candidate.get());
    return std::none_of(env.seeds.begin(), env.seeds.end(),
                        [&](const FxSymbolEnv::Seed &seed) { return seed.name == var->GetName(); });
  }
  if (candidate->GetKind() == Kind::Constant) {
    return false;
  }
  const auto *binary = static_cast<const fxrt::ir::SymbolicBinaryOp *>(candidate.get());
  return HasUnseededShapeSymbol(binary->getLHS(), env) || HasUnseededShapeSymbol(binary->getRHS(), env);
}

bool HasResolvableStride(const TensorRef &ref) {
  const auto &shape = LogicalShape(ref);
  if (ref.stride.size() != shape.size()) {
    return false;
  }
  if (!ref.stride_exprs.empty() && ref.stride_exprs.size() != shape.size()) return false;
  for (size_t i = 0; i < ref.stride.size(); ++i) {
    if (ref.stride[i] < 0 && (ref.stride_exprs.empty() || ref.stride_exprs[i] == nullptr)) return false;
  }
  return true;
}

bool EnsureUnpackedTensor(const TensorRef &input, const std::string &tag, size_t index,
                          LoweringContext &context, TensorRef *output, std::string *error) {
  *output = input;
  if (!IsDenseTensorRef(*output) && !HasResolvableStride(*output) && !output->proven_dense) {
    if (error != nullptr) {
      *error = "torch fallback: non-dense input requires a non-negative stride or a bound expression";
    }
    return false;
  }
  if (output->HasUnpackedPair()) {
    return true;
  }
  uint64_t shape_symbol = 0;
  uint64_t address_symbol = 0;
  auto *split = GetOrCreateSplitDataTensor(context, input, tag, index, &shape_symbol, &address_symbol);
  if (split == nullptr) {
    if (error != nullptr) {
      *error = "torch fallback: failed to unpack a tensor input";
    }
    return false;
  }
  output->unpacked.shape = TensorEndpoint{split, 0, shape_symbol};
  output->unpacked.addr = TensorEndpoint{split, 1, address_symbol};
  return true;
}

bool CollectFallbackInputs(const fxrt::ir::NodePtr &node, const std::string &tag,
                           LoweringContext &context, FallbackInputs *inputs, std::string *error,
                           size_t begin = 1U) {
  std::function<bool(const fxrt::ir::NodePtr &, TorchFallbackArgument *)> collect_node;
  collect_node = [&](const fxrt::ir::NodePtr &value_node, TorchFallbackArgument *argument) {
    const auto value = value_node == nullptr ? nullptr : value_node->output;
    argument->value = value;
    if (value != nullptr && value->IsTensor()) {
      TensorRef resolved;
      if (!ResolveProducerTensor(value_node, context, &resolved, error)) return false;
      TensorRef unpacked;
      if (!EnsureUnpackedTensor(resolved, "torch_fallback_" + tag, inputs->tensors.size(), context,
                                &unpacked, error)) return false;
      argument->tensor_index = inputs->tensors.size();
      const auto tensor = value->ToTensor();
      if (tensor != nullptr) {
        argument->tensor_format = static_cast<uint32_t>(tensor->Format());
        argument->ir_storage_sizes = tensor->StorageShape();
      }
      argument->ir_logical_shape = LogicalShape(unpacked);
      argument->ir_stride = unpacked.stride;
      inputs->tensors.push_back(std::move(unpacked));
      return true;
    }
    if (value != nullptr && value->IsTuple()) {
      const auto &tuple = value->ToTuple();
      argument->elements.resize(tuple->Size());
      for (size_t i = 0; i < tuple->Size(); ++i) {
        if (value_node != nullptr && i < value_node->inputs.size()) {
          if (!collect_node(value_node->inputs[i], &argument->elements[i])) return false;
        } else {
          argument->elements[i].value = (*tuple)[i];
        }
      }
    }
    return true;
  };
  for (size_t i = begin; i < node->inputs.size(); ++i) {
    const auto &input_node = node->inputs[i];
    if (input_node == nullptr || input_node->output == nullptr) {
      if (error != nullptr) {
        *error = "torch fallback: null ordered argument at index " + std::to_string(i - begin);
      }
      return false;
    }
    TorchFallbackArgument argument;
    if (!collect_node(input_node, &argument)) return false;
    inputs->arguments.push_back(std::move(argument));
  }
  return true;
}

bool WireFallbackInfer(const fxrt::ir::NodePtr &node, const std::string &tag,
                       const std::vector<TensorRef> &inputs, LoweringContext &context,
                       const ge::TensorDesc &output_desc, FallbackOutput *output, std::string *error,
                       const std::string &qualified_name = {},
                       const TorchFallbackCall *call = nullptr) {
  // Prefer deferred allocation for every functional Torch fallback. Capture-time
  // FakeTensor / SymInt templates (including specialized constants and expressions
  // like 2*((s72//2))) frequently disagree with runtime sizes after flash-comm
  // gathers; GE will not realloc once ExecuteCustomOpWithInferShape pins a
  // template. Out=/inplace schemas cannot use deferred, so they keep InferShape.
  (void)qualified_name;
  const bool out_schema = call != nullptr && TorchFallbackPlanUsesOutSchema(*call);
  if (!out_schema) {
    output->deferred_shape = true;
    return true;
  }
  bool runtime_shape = false;
  if (node != nullptr && node->output != nullptr && node->output->IsTensor() && context.symbol_env != nullptr) {
    const auto tensor = node->output->ToTensor();
    if (tensor != nullptr && tensor->HasSymbolicShape()) {
      const auto &symbolic_shape = tensor->GetSymbolicShape();
      runtime_shape = std::any_of(symbolic_shape.begin(), symbolic_shape.end(), [&](const auto &dim) {
        return HasUnseededShapeSymbol(dim, *context.symbol_env);
      });
    }
  }
  if (runtime_shape) {
    output->deferred_shape = true;
    return true;
  }
  auto rule = BuildOutputSymbolicShapeRule(node, context.symbol_env, "torch_fallback");
  if (rule == nullptr) {
    const auto shape = ShapeOfNode(node);
    if (shape.empty() || std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim < 0; })) {
      output->deferred_shape = true;
      return true;
    }
    if (error != nullptr) {
      *error = "torch fallback: output shape must be known from FakeTensor/SymInt metadata";
    }
    return false;
  }
  const FxShapeRule *rule_pointer = rule.get();
  context.built->shape_rules.push_back(std::move(rule));
  auto *rule_const = AddConstNode(context.main, "const_torch_fallback_rule_" + tag,
                                  &rule_pointer, sizeof(rule_pointer), false, *context.sym_gen,
                                  FxrtConstType::kVoidPtr);
  auto infer_function = GetFxExprInferShapeFunc();
  auto *infer_const = AddConstNode(context.main, "const_torch_fallback_infer_" + tag,
                                   &infer_function, sizeof(infer_function), false, *context.sym_gen,
                                   FxrtConstType::kKernelFuncPtr);
  output->infer = AddInferShapeNode(context.main, tag, inputs, output_desc, ge::TensorDesc(),
                                    rule_const, infer_const, *context.sym_gen, &output->shape_symbol);
  output->shape_rule_infer = output->infer;
  if (context.bind_symbols != nullptr) {
    AddControlEdge(context.main, context.bind_symbols, output->infer);
  }
  context.kernel_nodes->insert(context.kernel_nodes->end(), {rule_const, infer_const, output->infer});
  return true;
}

struct FallbackLaunchInputs {
  std::vector<fx_converter::Node *> nodes;
  std::vector<uint64_t> symbols;
};

bool BuildFallbackInputTensors(const std::string &tag, const std::vector<TensorRef> &inputs,
                               LoweringContext &context, FallbackLaunchInputs *built) {
  built->nodes.reserve(inputs.size());
  built->symbols.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    TensorRef input = inputs[i];
    if (!EnsureStridedTensorRefBoxed(&input, context.main, context.built,
                                     "torch_fallback_input_" + tag + "_" + std::to_string(i),
                                     *context.sym_gen, context.kernel_nodes) ||
        input.producer == nullptr) {
      return false;
    }
    built->nodes.push_back(input.producer);
    built->symbols.push_back(input.symbol);
  }
  return true;
}

// Compute inputs keep real tensor descriptors. Allocator, stream and the custom-op
// handler are trailing inputs with empty descriptors. Deferred ExecuteCustomOp has
// no template-shape inputs, and must not publish a concrete FakeTensor size: GE
// copies that nbytes onto the output and then skips the L1 malloc.
FxNodeSpecPtr MakeFallbackLaunchDesc(const std::string &tag, const TorchFallbackCall &call,
                                     const std::vector<TensorRef> &inputs,
                                     const std::vector<FallbackOutput> &outputs, bool deferred_shape,
                                     size_t template_input_begin,
                                     const std::vector<ge::TensorDesc> &shape_input_descs) {
  const ge::TensorDesc empty_td;
  auto desc = std::make_shared<FxNodeSpec>(
      "torch_fallback_" + tag, deferred_shape ? "FxTorchFallback" : "FxTorchFallbackWithInferShape");
  for (size_t i = 0; i < inputs.size(); ++i) {
    desc->AddInputDesc(MakeNdTensorDesc(LogicalShape(inputs[i]), inputs[i].dtype));
  }
  for (const auto &shape_desc : shape_input_descs) {
    desc->AddInputDesc(shape_desc);
  }
  for (size_t i = inputs.size() + shape_input_descs.size(); i < template_input_begin; ++i) {
    desc->AddInputDesc(empty_td);
  }
  for (size_t i = 0; !deferred_shape && i < outputs.size(); ++i) {
    desc->AddInputDesc(
        MakeNdTensorDesc(call.output_shapes[i], static_cast<ge::DataType>(call.output_dtypes[i])));
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (deferred_shape) {
      desc->AddOutputDesc(ge::TensorDesc(ge::Shape(), ge::FORMAT_ND,
                                           static_cast<ge::DataType>(call.output_dtypes[i])));
    } else {
      desc->AddOutputDesc(
          MakeNdTensorDesc(call.output_shapes[i], static_cast<ge::DataType>(call.output_dtypes[i])));
    }
  }
  if (call.shape_input_count != 0U) {
    (void)desc->SetAttr("fx_fallback_shape_input_begin", static_cast<int64_t>(call.tensor_count));
    (void)desc->SetAttr("fx_fallback_shape_input_count", static_cast<int64_t>(call.shape_input_count));
  }
  return desc;
}

bool WireFallbackLaunchNodes(const FallbackLaunchInputs &inputs, fx_converter::Node *find_node, uint64_t find_symbol,
                             fx_converter::Node *attached_stream_provider, uint64_t attached_stream_provider_symbol,
                             const std::vector<fx_converter::Node *> &context_inputs,
                             bool deferred_shape,
                             size_t template_input_begin, std::vector<FallbackOutput> &outputs,
                             LoweringContext &context, fx_converter::Node *launch, std::vector<uint64_t> *output_symbols) {
  std::vector<uint64_t> symbols;
  symbols.reserve(inputs.symbols.size() + kCustomOpExtraInputs);
  std::copy(inputs.symbols.begin(), inputs.symbols.end(), std::back_inserter(symbols));
  symbols.push_back(context.rt->s_allocator);
  symbols.push_back(context.rt->s_stream);
  symbols.push_back(find_symbol);
  symbols.push_back(attached_stream_provider_symbol);
  for (auto *input : context_inputs) {
    symbols.push_back(input->GetExtendInfo()->GetOutputSymbol(0U));
  }
  if (!deferred_shape) {
    std::transform(outputs.begin(), outputs.end(), std::back_inserter(symbols),
                   [](const FallbackOutput &output) { return output.shape_symbol; });
  }
  output_symbols->reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    output_symbols->push_back((*context.sym_gen)++);
  }
  WireSymbols(launch, symbols, *output_symbols);
  for (size_t i = 0; i < inputs.nodes.size(); ++i) {
    AddDataEdge(context.main, inputs.nodes[i], 0, launch, static_cast<int32_t>(i));
  }
  const int32_t tensor_count = static_cast<int32_t>(inputs.nodes.size());
  AddDataEdge(context.main, context.rt->select_l2, 0, launch, tensor_count);
  AddDataEdge(context.main, context.rt->split_streams, 0, launch, tensor_count + 1);
  AddDataEdge(context.main, find_node, 0, launch, tensor_count + 2);
  AddDataEdge(context.main, attached_stream_provider, 0, launch, tensor_count + 3);
  for (size_t i = 0; i < context_inputs.size(); ++i) {
    AddDataEdge(context.main, context_inputs[i], 0, launch, tensor_count + 4 + static_cast<int32_t>(i));
  }
  // Symbolic scalar args evaluate the shared symbol env. BindSymbolEnv must finish first.
  if (context.bind_symbols != nullptr) {
    AddControlEdge(context.main, context.bind_symbols, launch);
  }
  if (!deferred_shape) {
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i].infer == nullptr) {
        return false;
      }
      AddDataEdge(context.main, outputs[i].infer, 0, launch, static_cast<int32_t>(template_input_begin + i));
    }
  }
  context.kernel_nodes->push_back(launch);
  return true;
}

void SplitFallbackOutputs(const std::string &tag, fx_converter::Node *launch, const std::vector<uint64_t> &output_symbols,
                          LoweringContext &context, std::vector<FallbackOutput> *outputs) {
  const ge::TensorDesc empty_td;
  for (size_t i = 0; i < outputs->size(); ++i) {
    auto split_desc = std::make_shared<FxNodeSpec>("split_fx_fallback_" + tag + "_" + std::to_string(i),
                                                   "SplitDataTensor");
    split_desc->AddInputDesc(empty_td);
    split_desc->AddInputDesc(empty_td);
    split_desc->AddOutputDesc(empty_td);
    split_desc->AddOutputDesc(empty_td);
    auto *split = AddNode(context.main, split_desc);
    const uint64_t shape_sym = (*context.sym_gen)++;
    const uint64_t addr_sym = (*context.sym_gen)++;
    WireSymbols(split, {output_symbols[i], context.rt->s_allocator}, {shape_sym, addr_sym});
    AddDataEdge(context.main, launch, static_cast<int32_t>(i), split, 0);
    AddDataEdge(context.main, context.rt->select_l2, 0, split, 1);
    context.kernel_nodes->push_back(split);
    FallbackOutput &output = (*outputs)[i];
    output.infer = split;
    output.shape_symbol = shape_sym;
    output.source_alloc = split;
    output.source_address_symbol = addr_sym;
    if (output.output_data == nullptr) {
      output.alloc = split;
      output.address_symbol = addr_sym;
    }
  }
}

fx_converter::Node *WireFallbackLaunch(const std::string &tag, const std::shared_ptr<TorchFallbackCall> &call,
                                 const std::vector<TensorRef> &inputs, LoweringContext &context,
                                 std::vector<FallbackOutput> &outputs) {
  if (context.built == nullptr || call == nullptr) {
    return nullptr;
  }
  FallbackLaunchInputs inputs_built;
  if (!BuildFallbackInputTensors(tag, inputs, context, &inputs_built)) {
    return nullptr;
  }
  if (call->shape_input_count != 0U) {
    if (context.graph_input_datas == nullptr || context.graph_input_datas->size() < call->shape_input_count) {
      return nullptr;
    }
    for (size_t i = 0U; i < call->shape_input_count; ++i) {
      auto *feed = (*context.graph_input_datas)[i];
      if (feed == nullptr || feed->GetExtendInfo() == nullptr) {
        return nullptr;
      }
      ge::TensorDesc td;
      if (!feed->CopyOutputDesc(0U, &td)) {
        return nullptr;
      }
      call->shape_input_shapes.push_back(td.GetShape().GetDims());
      call->shape_input_dtypes.push_back(static_cast<int32_t>(td.GetDataType()));
      inputs_built.nodes.push_back(feed);
      inputs_built.symbols.push_back(feed->GetExtendInfo()->GetOutputSymbol(0U));
    }
  }
  TorchFallbackCall *call_ptr = call.get();
  auto *call_const = AddConstNode(context.main, "const_fx_fallback_call_" + tag, &call_ptr, sizeof(call_ptr), false,
                                  *context.sym_gen);
  if (call_const == nullptr || call_const->GetExtendInfo() == nullptr) {
    return nullptr;
  }
  const uint64_t call_symbol = (*context.sym_gen)++;
  call_const->GetExtendInfo()->SetOutputSymbol(0U, call_symbol);
  context.kernel_nodes->push_back(call_const);
  void *null_attached_stream_provider = nullptr;
  auto *attached_stream_provider = AddConstNode(context.main, "const_fx_fallback_attached_stream_provider_" + tag,
                                                 &null_attached_stream_provider,
                                                 sizeof(null_attached_stream_provider), false, *context.sym_gen,
                                                 FxrtConstType::kVoidPtr);
  const uint64_t attached_stream_provider_symbol = (*context.sym_gen)++;
  attached_stream_provider->GetExtendInfo()->SetOutputSymbol(0U, attached_stream_provider_symbol);
  context.kernel_nodes->push_back(attached_stream_provider);
  const bool deferred_shape = call->deferred_shape;
  // Match GE's ExecuteCustomOp context layout; unknown node and core IDs use -1.
  const uint32_t root_graph_id = 0U;
  const int64_t unknown = -1;
  std::vector<fx_converter::Node *> context_inputs = {
      AddConstNode(context.main, "const_fx_fallback_root_graph_id_" + tag, &root_graph_id,
                   sizeof(root_graph_id), false, *context.sym_gen, FxrtConstType::kUInt32T),
      AddConstNode(context.main, "const_fx_fallback_node_id_" + tag, &unknown,
                   sizeof(unknown), false, *context.sym_gen, FxrtConstType::kInt64T),
      AddConstNode(context.main, "const_fx_fallback_vector_core_num_" + tag, &unknown,
                   sizeof(unknown), false, *context.sym_gen, FxrtConstType::kInt64T)};
  for (auto *input : context_inputs) {
    if (input == nullptr || input->GetExtendInfo() == nullptr) {
      return nullptr;
    }
    context.kernel_nodes->push_back(input);
  }
  const size_t template_input_begin = inputs_built.nodes.size() + kCustomOpExtraInputs;
  std::vector<ge::TensorDesc> shape_input_descs;
  shape_input_descs.reserve(call->shape_input_count);
  for (size_t i = 0U; i < call->shape_input_count; ++i) {
    ge::TensorDesc shape_td;
    if ((*context.graph_input_datas)[i] == nullptr ||
        !(*context.graph_input_datas)[i]->CopyOutputDesc(0U, &shape_td)) {
      return nullptr;
    }
    shape_input_descs.push_back(shape_td);
  }
  auto desc = MakeFallbackLaunchDesc(tag, *call, inputs, outputs, deferred_shape, template_input_begin,
                                     shape_input_descs);
  auto *launch = AddNode(context.main, desc);
  std::vector<uint64_t> output_symbols;
  if (!WireFallbackLaunchNodes(inputs_built, call_const, call_symbol, attached_stream_provider,
                               attached_stream_provider_symbol, context_inputs, deferred_shape, template_input_begin, outputs,
                               context, launch, &output_symbols)) {
    return nullptr;
  }
  SplitFallbackOutputs(tag, launch, output_symbols, context, &outputs);
  return launch;
}

const TensorRef *FallbackAliasSource(const TorchFallbackCall *call, const std::vector<TensorRef> &inputs,
                                    const std::vector<std::pair<size_t, size_t>> &pairs,
                                    size_t output_index, const char *kind, std::string *error) {
  if (call == nullptr) return nullptr;
  for (const auto &pair : pairs) {
    if (pair.second != output_index) continue;
    if (pair.first >= call->arguments.size()) {
      if (error != nullptr) *error = std::string("torch fallback: ") + kind + " argument index is out of range";
      return nullptr;
    }
    const size_t tensor_index = call->arguments[pair.first].tensor_index;
    if (tensor_index == kNoTorchFallbackTensor || tensor_index >= inputs.size() ||
        !inputs[tensor_index].HasUnpackedPair()) {
      if (error != nullptr) *error = std::string("torch fallback: ") + kind + " input has no storage pair";
      return nullptr;
    }
    return &inputs[tensor_index];
  }
  return nullptr;
}

bool ApplyFallbackAliasLayout(const FallbackOutputSpec &spec, const TorchFallbackCall &call,
                              const std::vector<TensorRef> &inputs, size_t output_index,
                              TensorRef *output, std::string *error) {
  const TensorRef *source = FallbackAliasSource(&call, inputs, call.ref_pairs, output_index, "ref", error);
  if (error != nullptr && !error->empty()) return false;
  if (source == nullptr) {
    source = FallbackAliasSource(&call, inputs, call.read_alias_pairs, output_index, "read-alias", error);
    if (error != nullptr && !error->empty()) return false;
  }
  if (source == nullptr) return true;

  const bool write_alias = std::any_of(call.ref_pairs.begin(), call.ref_pairs.end(),
                                       [output_index](const auto &pair) { return pair.second == output_index; });
  if (write_alias) {
    output->shape = source->shape;
    output->storage_shape = source->storage_shape;
    output->logical_shape = source->logical_shape;
    output->logical_shape_exprs = source->logical_shape_exprs;
    output->dtype = source->dtype;
    output->stride = source->stride;
    output->stride_exprs = source->stride_exprs;
    output->proven_dense = source->proven_dense;
    output->storage_offset = source->storage_offset;
    output->storage_offset_expr = source->storage_offset_expr;
    output->runtime_storage_offset = source->runtime_storage_offset;
    output->format = source->format;
    output->feed_index = source->feed_index;
    output->allocation = source->allocation.Valid() ? source->allocation : source->unpacked.addr;
    output->can_adopt_for_model_output = false;
    return true;
  }

  output->storage_shape = spec.shape;
  output->logical_shape = spec.shape;
  output->logical_shape_exprs.clear();
  const auto value = spec.shape_node == nullptr ? nullptr : spec.shape_node->output;
  const auto tensor = value != nullptr && value->IsTensor() ? value->ToTensor() : nullptr;
  if (tensor != nullptr && tensor->HasSymbolicShape()) {
    if (call.symbol_env == nullptr ||
        !SymbolicShapeExprsOfNode(spec.shape_node, *call.symbol_env, &output->logical_shape_exprs) ||
        output->logical_shape_exprs.size() != spec.shape.size() ||
        tensor->GetSymbolicShape().size() != spec.shape.size()) {
      if (error != nullptr) *error = "torch fallback: read-alias output has unresolved logical shape";
      return false;
    }
  }
  if (std::any_of(spec.shape.begin(), spec.shape.end(), [](int64_t dim) { return dim < 0; }) &&
      output->logical_shape_exprs.size() != spec.shape.size()) {
    if (error != nullptr) *error = "torch fallback: read-alias output has unresolved logical shape";
    return false;
  }
  output->stride.assign(spec.shape.size(), -1);
  output->stride_exprs.clear();
  output->proven_dense = true;
  output->storage_offset = source->storage_offset;
  output->storage_offset_expr = source->storage_offset_expr;
  output->runtime_storage_offset = source->runtime_storage_offset;
  output->feed_index = source->feed_index;
  output->format = source->format;
  output->allocation = source->allocation.Valid() ? source->allocation : source->unpacked.addr;
  output->can_adopt_for_model_output = false;
  return true;
}

bool FillFallbackResult(const TorchFallbackCall &call, const std::vector<TensorRef> &inputs,
                        const std::vector<int64_t> &shape, ge::DataType dtype, const FallbackOutputSpec &spec,
                        const FallbackOutput &stage, fx_converter::Node *launch,
                        LoweringResult *result, std::string *error) {
  TensorRef output;
  output.producer = nullptr;
  output.symbol = 0U;
  output.shape = shape;
  output.dtype = dtype;
  output.unpacked.shape = TensorEndpoint{stage.infer, 0, stage.shape_symbol};
  output.unpacked.addr = TensorEndpoint{stage.source_alloc != nullptr ? stage.source_alloc : stage.alloc,
                                        stage.source_alloc != nullptr ? 1 : 0,
                                        stage.source_address_symbol != 0 ? stage.source_address_symbol
                                                                         : stage.address_symbol};
  output.can_adopt_for_model_output = true;
  if (!ApplyFallbackAliasLayout(spec, call, inputs, 0U, &output, error)) return false;
  result->outputs = {std::move(output)};
  result->launch = launch;
  result->prepare = launch;
  result->mem_to_frees.push_back(stage.source_alloc != nullptr ? stage.source_alloc : stage.alloc);
  // GE resolves this registered proto to validate output-ref-input port names.
  result->cni.op_type = call.proto_type;
  result->cni.cni_io = CniIoStyle::kIndexedFrom0;
  result->cni.behavior = nullptr;
  for (const auto &input : inputs) {
    result->cni.input_shapes.push_back(PhysicalShape(input));
    result->cni.input_dtypes.push_back(static_cast<int32_t>(input.dtype));
  }
  for (size_t i = 0U; i < call.shape_input_shapes.size(); ++i) {
    result->cni.input_shapes.push_back(call.shape_input_shapes[i]);
    result->cni.input_dtypes.push_back(call.shape_input_dtypes[i]);
  }
  result->cni.output_shape = shape;
  result->cni.output_dtype = static_cast<int32_t>(dtype);
  // ComputeNodeInfo describes the GE custom-op launch ABI. InferShape has a
  // separate, shape-producing signature and must not share this CNI (in
  // particular for multi-output calls where the launch has N outputs but each
  // InferShape node has exactly one).
  result->cni.compute_nodes = {launch};
  if (stage.shape_rule_infer != nullptr) {
    result->cni.compute_nodes.push_back(stage.shape_rule_infer);
  }
  return true;
}

bool MakeFallbackTensorRef(const FallbackOutputSpec &spec, const FallbackOutput &stage,
                           const TorchFallbackCall &call, const std::vector<TensorRef> &inputs,
                           size_t output_index, TensorRef *output, std::string *error) {
  if (output == nullptr) return false;
  output->producer = nullptr;
  output->symbol = 0U;
  output->shape = spec.shape;
  output->dtype = spec.dtype;
  output->unpacked.shape = TensorEndpoint{stage.infer, 0, stage.shape_symbol};
  output->unpacked.addr = TensorEndpoint{stage.source_alloc != nullptr ? stage.source_alloc : stage.alloc,
                                         stage.source_alloc != nullptr ? 1 : 0,
                                         stage.source_address_symbol != 0 ? stage.source_address_symbol
                                                                          : stage.address_symbol};
  output->can_adopt_for_model_output = true;
  return ApplyFallbackAliasLayout(spec, call, inputs, output_index, output, error);
}

// Multi-output producers publish every slot in slot order; the builder records the
// vector and later tuple_getitem nodes alias into it natively (same contract as
// catalog multi-output ops like TopKV2).
bool FillFallbackMultiResult(const TorchFallbackCall &call, const std::vector<TensorRef> &inputs,
                             const std::vector<FallbackOutputSpec> &specs,
                             const std::vector<FallbackOutput> &stages, fx_converter::Node *launch,
                             LoweringResult *result, std::string *error) {
  result->outputs.reserve(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    TensorRef output;
    if (!MakeFallbackTensorRef(specs[i], stages[i], call, inputs, i, &output, error)) {
      result->outputs.clear();
      return false;
    }
    result->outputs.push_back(std::move(output));
  }
  result->launch = launch;
  result->prepare = launch;
  for (size_t i = 0; i < stages.size(); ++i) {
    result->mem_to_frees.push_back(stages[i].source_alloc != nullptr ? stages[i].source_alloc : stages[i].alloc);
  }
  // GE resolves this registered proto to validate output-ref-input port names.
  result->cni.op_type = call.proto_type;
  result->cni.cni_io = CniIoStyle::kIndexedFrom0;
  result->cni.behavior = nullptr;
  for (const auto &input : inputs) {
    result->cni.input_shapes.push_back(PhysicalShape(input));
    result->cni.input_dtypes.push_back(static_cast<int32_t>(input.dtype));
  }
  for (size_t i = 0U; i < call.shape_input_shapes.size(); ++i) {
    result->cni.input_shapes.push_back(call.shape_input_shapes[i]);
    result->cni.input_dtypes.push_back(call.shape_input_dtypes[i]);
  }
  result->cni.output_shape = specs[0].shape;
  result->cni.output_dtype = static_cast<int32_t>(specs[0].dtype);
  for (size_t i = 0; i < specs.size(); ++i) {
    result->cni.output_shapes.push_back(specs[i].shape);
    result->cni.output_dtypes.push_back(static_cast<int32_t>(specs[i].dtype));
  }
  result->cni.compute_nodes.push_back(launch);
  for (const auto &stage : stages) {
    if (stage.shape_rule_infer != nullptr) {
      result->cni.compute_nodes.push_back(stage.shape_rule_infer);
    }
  }
  return true;
}

// Shared tail of a tuple-output fallback producer: the call is created and the
// per-slot specs carry the import-stamped FakeTensor metadata, so all that is
// left is wiring one stage per slot and publishing every slot.
bool LowerFallbackTupleOutputs(const std::string &tag, const std::shared_ptr<TorchFallbackCall> &call,
                               const std::vector<TensorRef> &inputs,
                               const std::vector<FallbackOutputSpec> &specs, const std::string &op_type,
                               LoweringContext &context, LoweringResult *result, std::string *error) {
  for (const auto &spec : specs) {
    call->output_shapes.push_back(spec.shape);
  }
  context.built->torch_fallback_calls.push_back(call);
  std::vector<FallbackOutput> stages(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    const std::string output_tag = tag + "_o" + std::to_string(i);
    const auto output_desc = MakeNdTensorDesc(specs[i].shape, specs[i].dtype);
    if (!WireFallbackInfer(specs[i].shape_node, output_tag, inputs, context, output_desc, &stages[i],
                           error, op_type, call.get())) {
      return false;
    }
  }
  call->deferred_shape =
      std::any_of(stages.begin(), stages.end(), [](const FallbackOutput &stage) { return stage.deferred_shape; });
  auto *launch = WireFallbackLaunch(tag, call, inputs, context, stages);
  if (launch == nullptr) {
    if (error != nullptr) {
      *error = "torch fallback: failed to wire FxTorchFallback";
    }
    return false;
  }
  return FillFallbackMultiResult(*call, inputs, specs, stages, launch, result, error);
}

std::vector<int32_t> FallbackOutputDtypes(const std::vector<FallbackOutputSpec> &specs) {
  std::vector<int32_t> output_dtypes;
  output_dtypes.reserve(specs.size());
  std::transform(specs.begin(), specs.end(), std::back_inserter(output_dtypes),
                 [](const FallbackOutputSpec &spec) { return static_cast<int32_t>(spec.dtype); });
  return output_dtypes;
}

bool LowerCustomCallTuple(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &context,
                          LoweringResult *result, std::string *error) {
  std::string qualified_name;
  std::string overload_name;
  if (!ReadQualifiedName(node, &qualified_name, &overload_name, error)) {
    return false;
  }
  FallbackInputs inputs;
  if (!CollectFallbackInputs(node, tag, context, &inputs, error)) {
    return false;
  }
  std::vector<FallbackOutputSpec> specs;
  if (!CollectFallbackOutputs(node, &specs, error) || specs.empty()) {
    if (error != nullptr && error->empty()) {
      *error = "torch fallback: tuple output requires a fixed tuple/list of at least one Tensor";
    }
    return false;
  }
  auto call = CreateTorchFallbackCall(qualified_name, overload_name, std::move(inputs.arguments), context.symbol_env,
                                      FallbackOutputDtypes(specs), ReadFallbackReadAliases(node, specs.size()),
                                      ReadPythonOutputDenseFlags(node, specs.size(), "fxrt_fallback_fresh_dense"),
                                      context.built->fallback_session, error);
  if (call == nullptr) {
    return false;
  }
  return LowerFallbackTupleOutputs(tag, call, inputs.tensors, specs, qualified_name, context, result, error);
}

bool LowerCustomCallTensor(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &context,
                           LoweringResult *result, std::string *error) {
  std::string qualified_name;
  std::string overload_name;
  if (!ReadQualifiedName(node, &qualified_name, &overload_name, error)) {
    return false;
  }
  FallbackInputs inputs;
  if (!CollectFallbackInputs(node, tag, context, &inputs, error)) {
    return false;
  }
  const auto output_shape = ShapeOfNode(node);
  const auto output_dtype = DtypeOfNode(node);
  if (node->output == nullptr || !node->output->IsTensor()) {
    if (error != nullptr) {
      *error = "torch fallback: output must be a Tensor or a fixed tuple/list of Tensors";
    }
    return false;
  }
  auto call = CreateTorchFallbackCall(qualified_name, overload_name, std::move(inputs.arguments), context.symbol_env,
                                      {static_cast<int32_t>(output_dtype)}, ReadFallbackReadAliases(node, 1U),
                                      ReadPythonOutputDenseFlags(node, 1U, "fxrt_fallback_fresh_dense"),
                                      context.built->fallback_session, error);
  if (call == nullptr) {
    return false;
  }
  call->output_shapes.push_back(output_shape);
  context.built->torch_fallback_calls.push_back(call);
  FallbackOutput output;
  const auto output_desc = MakeNdTensorDesc(output_shape, output_dtype);
  if (!WireFallbackInfer(node, tag, inputs.tensors, context, output_desc, &output, error, qualified_name,
                         call.get())) {
    return false;
  }
  call->deferred_shape = output.deferred_shape;
  std::vector<FallbackOutput> output_stages{output};
  auto *launch = WireFallbackLaunch(tag, call, inputs.tensors, context, output_stages);
  output = output_stages[0];
  FallbackOutputSpec output_spec{node, output_shape, output_dtype};
  return FillFallbackResult(*call, inputs.tensors, output_shape, output_dtype, output_spec, output,
                            launch, result, error);
}

class TorchFallbackOpLowering final : public OpLowering {
 public:
  bool MayWriteInputs() const override { return true; }
  InputLayoutKind InputLayout(const fxrt::ir::NodePtr &, size_t) const override {
    return InputLayoutKind::kStridedTensor;
  }
  const char *Name() const override { return "TorchFallback"; }

  std::vector<std::string> SupportedLogicalOps() const override { return {"custom_call"}; }

  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *ctx = nullptr) const override {
    (void)ctx;
    return node != nullptr && node->op == fxrt::ops::Op_custom_call;
  }

  bool Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &context,
             LoweringResult *result, std::string *error) override {
    if (result == nullptr || !ValidateLoweringContext(context, false, "torch fallback", error)) {
      return false;
    }
    // Tensor[] producers publish every slot; the builder aliases tuple_getitem
    // consumers into the recorded slots natively.
    if (node != nullptr && node->output != nullptr && node->output->IsTuple()) {
      return LowerCustomCallTuple(node, tag, context, result, error);
    }
    return LowerCustomCallTensor(node, tag, context, result, error);
  }
};

constexpr int kPythonCallPriority = 9990;

bool ReadPythonCallTarget(const fxrt::ir::NodePtr &node, std::string *module_name,
                          std::string *func_name, std::string *error) {
  if (node == nullptr || node->inputs.size() < 2U || node->inputs[0] == nullptr ||
      node->inputs[1] == nullptr || node->inputs[0]->output == nullptr ||
      node->inputs[1]->output == nullptr || !node->inputs[0]->output->IsString() ||
      !node->inputs[1]->output->IsString()) {
    if (error != nullptr) {
      *error = "python call: missing module_name / func_name string inputs";
    }
    return false;
  }
  *module_name = node->inputs[0]->output->ToString();
  *func_name = node->inputs[1]->output->ToString();
  return !module_name->empty() && !func_name->empty();
}

bool LowerPythonCallProducer(const fxrt::ir::NodePtr &producer, const std::string &tag,
                             LoweringContext &context, LoweringResult *result, std::string *error) {
  std::string module_name;
  std::string func_name;
  if (!ReadPythonCallTarget(producer, &module_name, &func_name, error)) {
    return false;
  }
  FallbackInputs inputs;
  if (!CollectFallbackInputs(producer, tag, context, &inputs, error, 2U)) {
    return false;
  }
  std::vector<FallbackOutputSpec> specs;
  if (!CollectFallbackOutputs(producer, &specs, error) || specs.empty()) {
    return false;
  }
  std::vector<int32_t> output_dtypes;
  output_dtypes.reserve(specs.size());
  std::transform(specs.begin(), specs.end(), std::back_inserter(output_dtypes),
                 [](const FallbackOutputSpec &spec) { return static_cast<int32_t>(spec.dtype); });
  auto call = CreatePythonCall(module_name, func_name, std::move(inputs.arguments), context.symbol_env,
                               std::move(output_dtypes), context.built->fallback_session, error);
  if (call == nullptr) {
    return false;
  }
  call->python_output_dense.reserve(specs.size());
  for (const auto &spec : specs) {
    call->python_output_dense.push_back(spec.python_dense);
  }
  const std::string op_type = module_name + "." + func_name;
  // Tuple producers publish every slot; the builder aliases tuple_getitem
  // consumers into the recorded slots natively.
  if (producer->output->IsTuple()) {
    return LowerFallbackTupleOutputs(tag, call, inputs.tensors, specs, op_type, context, result, error);
  }
  call->output_shapes.push_back(specs[0].shape);
  context.built->torch_fallback_calls.push_back(call);
  FallbackOutput output;
  const auto output_desc = MakeNdTensorDesc(specs[0].shape, specs[0].dtype);
  if (!WireFallbackInfer(specs[0].shape_node, tag, inputs.tensors, context, output_desc, &output, error,
                         op_type, call.get())) {
    return false;
  }
  call->deferred_shape = output.deferred_shape;
  std::vector<FallbackOutput> output_stages{output};
  auto *launch = WireFallbackLaunch(tag, call, inputs.tensors, context, output_stages);
  if (launch == nullptr) {
    if (error != nullptr) {
      *error = "python call: failed to wire FxTorchFallback";
    }
    return false;
  }
  output = output_stages[0];
  return FillFallbackResult(*call, inputs.tensors, specs[0].shape, specs[0].dtype, specs[0], output,
                            launch, result, error);
}

class PythonCallOpLowering final : public OpLowering {
 public:
  bool MayWriteInputs() const override { return true; }
  InputLayoutKind InputLayout(const fxrt::ir::NodePtr &, size_t) const override {
    return InputLayoutKind::kStridedTensor;
  }
  const char *Name() const override { return "PythonCall"; }

  std::vector<std::string> SupportedLogicalOps() const override { return {"python_call"}; }

  bool Match(const fxrt::ir::NodePtr &node, const LoweringContext *ctx = nullptr) const override {
    (void)ctx;
    return node != nullptr && node->op == fxrt::ops::Op_python_call;
  }

  bool Lower(const fxrt::ir::NodePtr &node, const std::string &tag, LoweringContext &context,
             LoweringResult *result, std::string *error) override {
    if (result == nullptr || !ValidateLoweringContext(context, false, "python call", error)) {
      return false;
    }
    return LowerPythonCallProducer(node, tag, context, result, error);
  }
};

}  // namespace
}  // namespace fx_converter

FX_REGISTER_OP_LOWERING(::fx_converter::kTorchFallbackPriority,
                        std::make_unique<::fx_converter::TorchFallbackOpLowering>());
namespace {
static ::fx_converter::OpLoweringRegistrar _fx_python_call_lowering_reg(
    9990, std::make_unique<::fx_converter::PythonCallOpLowering>());
}
