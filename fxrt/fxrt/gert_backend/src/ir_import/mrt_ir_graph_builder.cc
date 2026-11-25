#include "ir_import/mrt_ir_graph_builder.h"

#include <algorithm>
#include <iterator>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "hardware/device.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "ops/op_def/ops_name.h"
#include "register/op_catalog_register.h"

namespace fx_converter {
namespace {

fxrt::ir::DataType ParseMetaDtype(const std::string &dtype) {
  if (dtype.empty()) {
    return fxrt::ir::DataType::Float32;
  }
  try {
    return fxrt::ir::DataType::FromString(dtype);
  } catch (...) {
    return fxrt::ir::DataType::Float32;
  }
}

fxrt::ir::ValuePtr MakeMetaTensorValue(const std::vector<int64_t> &shape,
                                      const std::vector<std::string> &sym_shape,
                                      const std::string &dtype = {}) {
  std::vector<int64_t> dims;
  if (!shape.empty()) {
    dims = shape;
  } else if (!sym_shape.empty()) {
    // Rank from symbolic dims; concrete sizes stay dynamic (-1).
    dims.assign(sym_shape.size(), -1);
  }
  // Empty shape AND empty sym_shape is a genuine rank-0 tensor (e.g. argmax without
  // keepdim, a full reduce): every real tensor export carries its shape, so an empty
  // list here can only mean 0-d. Spurious SymInt-scaffolding feeds are dropped earlier
  // by IsSpuriousEmptyFeed and never reach this builder.
  const fxrt::hardware::Device cpu{fxrt::hardware::DeviceType::CPU, 0};
  auto tensor = fxrt::ir::MakeIntrusive<fxrt::ir::Tensor>(dims, ParseMetaDtype(dtype), cpu);
  if (!sym_shape.empty()) {
    std::vector<fxrt::ir::SymbolicExprPtr> exprs;
    exprs.reserve(sym_shape.size());
    std::transform(sym_shape.begin(), sym_shape.end(), std::back_inserter(exprs),
                   [](const auto &s) -> fxrt::ir::SymbolicExprPtr {
                     if (s.rfind("c:", 0) == 0) {
                       return fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicConst>(std::stoll(s.substr(2)));
                     }
                     // Schema 5 also exports concrete dimensions as plain
                     // decimal strings. A dimension "1" is not a free symbol:
                     // losing that fact breaks squeeze/unsqueeze layout proofs.
                     if (!s.empty()) {
                       size_t consumed = 0U;
                       try {
                         const int64_t value = std::stoll(s, &consumed);
                         if (consumed == s.size()) {
                           return fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicConst>(value);
                         }
                       } catch (const std::exception &) {
                         // General symbolic expressions keep the existing channel.
                       }
                     }
                     return fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicVar>(s);
                   });
    tensor->SetSymbolicShape(exprs);
  }
  return fxrt::ir::MakeIntrusive<fxrt::ir::Value>(tensor);
}

fxrt::ir::MemoryFormat FormatFromCode(int32_t code) {
  // Negative is FORMAT_UNDEFINED. 0 is FORMAT_NCHW and must be kept.
  if (code < 0) {
    return fxrt::ir::FORMAT_ND;
  }
  return static_cast<fxrt::ir::MemoryFormat>(code);
}

void ApplyFormatCode(const fxrt::ir::ValuePtr &value, int32_t code) {
  if (value == nullptr || !value->IsTensor()) {
    return;
  }
  value->ToTensor()->SetFormat(FormatFromCode(code));
}

void ApplyTensorLayout(const IrNodeSpec &spec, const fxrt::ir::ValuePtr &value) {
  if (value == nullptr || !value->IsTensor()) {
    return;
  }
  const auto tensor = value->ToTensor();
  if (!spec.stride.empty()) tensor->SetStrides(spec.stride);
  if (!spec.storage_shape.empty()) tensor->SetStorageShape(spec.storage_shape);
  tensor->SetStorageOffset(spec.storage_offset);
  ApplyFormatCode(value, spec.format);
}

// Dynamo dynamic=True may export SymInt scaffolding as empty-shape float tensors.
// ModelV2 only feeds real tensor Data; drop those placeholders at import.
bool IsSpuriousEmptyFeed(const IrNodeSpec &n) {
  return n.shape.empty() && n.sym_shape.empty();
}

// The aten.item / _local_scalar_dense fold is how a Dynamo-lifted scalar placeholder
// is consumed; such a consumer does not make its operand a real tensor feed.
bool IsScalarFoldSpecOp(const std::string &op) {
  return op == "aten.item" || op == "aten._local_scalar_dense";
}

void CollectRealOpInputRefs(const IrValueSpec &value, std::unordered_set<std::string> *refs) {
  if (value.kind == IrValueSpec::Kind::kRef) {
    refs->insert(value.ref);
  }
  for (const auto &element : value.elements) {
    CollectRealOpInputRefs(element, refs);
  }
}

// Names referenced by a real consumer (compute op or graph return, excluding the
// scalar-fold ops above). An empty-shape feed whose name appears here is a genuine
// rank-0 tensor input (e.g. the rank-0 scales of npu_quantize) and must survive the
// spurious-feed drop; an empty-shape feed referenced nowhere else is Dynamo SymInt
// scaffolding folded into a constant at import.
std::unordered_set<std::string> CollectRealOpInputRefs(const IrGraphSpec &spec) {
  std::unordered_set<std::string> refs;
  for (const auto &n : spec.nodes) {
    if (n.kind == IrNodeKind::kInput || n.kind == IrNodeKind::kParameter || IsScalarFoldSpecOp(n.op)) {
      continue;
    }
    refs.insert(n.inputs.begin(), n.inputs.end());
    for (const auto &argument : n.arguments) {
      CollectRealOpInputRefs(argument, &refs);
    }
  }
  return refs;
}

bool TryMatchOp(const std::string &name, fxrt::ops::Op *out) {
  for (int i = 0; i < static_cast<int>(fxrt::ops::Op_End); ++i) {
    const auto cand = static_cast<fxrt::ops::Op>(i);
    if (name == fxrt::ops::ToStr(cand)) {
      *out = cand;
      return true;
    }
  }
  return false;
}

fxrt::ir::NodePtr MakeArgumentNode(const IrValueSpec &spec, const fxrt::ir::GraphPtr &graph,
                                  const std::unordered_map<std::string, fxrt::ir::NodePtr> &by_name) {
  if (spec.kind == IrValueSpec::Kind::kRef) {
    const auto it = by_name.find(spec.ref);
    return it == by_name.end() ? nullptr : it->second;
  }
  auto node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
  node->op = fxrt::ops::Op_End;
  fxrt::ir::ValuePtr value;
  switch (spec.kind) {
    case IrValueSpec::Kind::kInt:
      value = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(spec.int_value);
      break;
    case IrValueSpec::Kind::kDouble:
      value = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(spec.double_value);
      break;
    case IrValueSpec::Kind::kBool:
      value = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(spec.bool_value);
      break;
    case IrValueSpec::Kind::kString:
      value = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(std::string(spec.string_value));
      break;
    case IrValueSpec::Kind::kSymbol:
      value = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(
          fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicVar>(spec.string_value));
      break;
    case IrValueSpec::Kind::kTuple: {
      node->op = fxrt::ops::Op_make_tuple;
      std::vector<fxrt::ir::ValuePtr> elements;
      elements.reserve(spec.elements.size());
      for (const auto &element : spec.elements) {
        const auto element_node = MakeArgumentNode(element, graph, by_name);
        if (element_node == nullptr || element_node->output == nullptr) {
          return nullptr;
        }
        node->inputs.push_back(element_node);
        elements.push_back(element_node->output);
      }
      value = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(fxrt::ir::MakeIntrusive<fxrt::ir::Tuple>(std::move(elements)));
      break;
    }
    case IrValueSpec::Kind::kNone:
    default:
      value = fxrt::ir::MakeIntrusive<fxrt::ir::Value>();
      break;
  }
  node->output = value;
  graph->nodes.push_back(node);
  return node;
}

}  // namespace

static void AddInputOrParameterNode(const IrNodeSpec &n, const fxrt::ir::GraphPtr &graph,
                                    std::unordered_map<std::string, fxrt::ir::NodePtr> &by_name,
                                    const std::unordered_set<std::string> &real_refs) {
  if (IsSpuriousEmptyFeed(n) && real_refs.count(n.name) == 0) {
    return;
  }
  auto node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
  node->op = fxrt::ops::Op_End;
  node->output = MakeMetaTensorValue(n.shape, n.sym_shape, n.dtype);
  ApplyTensorLayout(n, node->output);
  for (const auto &attr : n.attrs) {
    const auto value = MakeArgumentNode(attr.second, graph, by_name);
    if (value != nullptr) node->attrs[attr.first] = value->output;
  }
  if (n.kind == IrNodeKind::kInput) {
    graph->inputs.push_back(node);
  } else {
    graph->parameters.push_back(node);
  }
  by_name[n.name] = node;
}

static void AddReturnNode(const IrNodeSpec &n, const fxrt::ir::GraphPtr &graph,
                          std::unordered_map<std::string, fxrt::ir::NodePtr> &by_name) {
  auto node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
  node->op = fxrt::ops::Op_return;
  // Multi-output graphs (TopKV2): NetOutput consumes N return tensors.
  for (const auto &in_name : n.inputs) {
    auto it = by_name.find(in_name);
    if (it != by_name.end()) {
      node->inputs.push_back(it->second);
    }
  }
  graph->nodes.push_back(node);
  by_name[n.name] = node;
}

static bool UsesOrderedArguments(fxrt::ops::Op op) {
  // Structural nodes (no catalog row) keep their hardwired ordered-arguments contract.
  if (op == fxrt::ops::Op_empty || op == fxrt::ops::Op_empty_strided || op == fxrt::ops::Op_compiled_kernel_mutation ||
      op == fxrt::ops::Op_python_call) {
    return true;
  }
  // Catalog ops opt in via CatalogRowBuilder::OrderedArguments: the normalized bag
  // drops None/bool/string operands and compacts Int/Double slots, so positional
  // behavior rules (ClassifyAt ordinals, string captures) are only meaningful
  // against the full schema layout the ordered channel carries.
  const fx_converter::OpCatalogEntry *entry =
      fx_converter::CatalogRegistry::Instance().Find(fxrt::ops::ToStr(op));
  return entry != nullptr && entry->ordered_arguments;
}

// A converter graph rewrite already stored the underlying inputs as arguments
// (data tensor plus integer tuples). The normalized bag has a single int_list,
// so an empty int_list with a tuple argument is rebuilt as those arguments.
static bool ArgumentsCarryTuple(const std::vector<IrValueSpec> &arguments) {
  return std::any_of(arguments.begin(), arguments.end(),
                      [](const IrValueSpec &spec) { return spec.kind == IrValueSpec::Kind::kTuple; });
}

static bool IsRewrittenTupleOp(fxrt::ops::Op op) {
  return op == fxrt::ops::Op_slice || op == fxrt::ops::Op_getitem_slice;
}

static bool PreferOrderedArguments(fxrt::ops::Op op, const IrNodeSpec &spec) {
  if (spec.arguments.empty()) {
    return false;
  }
  // custom_call / python_call keep scalars, None, and strings. Normalized inputs drop them.
  if (UsesOrderedArguments(op) || op == fxrt::ops::Op_custom_call) {
    return true;
  }
  return IsRewrittenTupleOp(op) && spec.int_list.empty() && ArgumentsCarryTuple(spec.arguments);
}

static void AddOrderedArguments(const IrNodeSpec &n, const fxrt::ir::GraphPtr &graph,
                                const std::unordered_map<std::string, fxrt::ir::NodePtr> &by_name,
                                const fxrt::ir::NodePtr &node, size_t begin = 0) {
  for (size_t i = begin; i < n.arguments.size(); ++i) {
    const auto input = MakeArgumentNode(n.arguments[i], graph, by_name);
    if (input == nullptr) {
      throw std::invalid_argument(n.op + " has unresolved ordered argument " + std::to_string(i));
    }
    node->inputs.push_back(input);
  }
}

static void AddNormalizedInputs(const IrNodeSpec &n,
                                std::unordered_map<std::string, fxrt::ir::NodePtr> &by_name,
                                const fxrt::ir::NodePtr &node) {
  for (const auto &in_name : n.inputs) {
    auto it = by_name.find(in_name);
    if (it == by_name.end()) {
      // The export routes a symbolic scalar attr (a Dynamo-lifted SymInt/SymFloat) to an
      // unknown_symbol_ name: it can never resolve, so fail here with the reason instead
      // of letting a stub tensor through to a cryptic arity error.
      if (in_name.find("unknown_symbol_") == 0) {
        throw std::invalid_argument(n.op + " has a symbolic scalar argument (" + in_name +
                                    ") that cannot be baked into an attr; keep the scalar a literal "
                                    "in the traced body");
      }
      auto stub = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
      stub->op = fxrt::ops::Op_End;
      stub->output = MakeMetaTensorValue({-1, -1}, {});
      by_name[in_name] = stub;
      it = by_name.find(in_name);
    }
    node->inputs.push_back(it->second);
  }
  if (!n.extra_ints.empty()) {
    // Ordered multi-Int channel (div_mod_scalar [x, scalar, mode]): the legacy axis field
    // keeps only the last Int operand, so rebuild every Int operand in argument order. A
    // single-Int node takes the same shape as the axis branch below, so both stay
    // equivalent there. A slot whose aligned extra_syms entry is non-empty is a symbolic
    // scalar (arange limit=s27): rebuild it as a Symbol Value like sym_axis does.
    for (size_t i = 0; i < n.extra_ints.size(); ++i) {
      auto int_node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
      int_node->op = fxrt::ops::Op_End;
      if (i < n.extra_syms.size() && !n.extra_syms[i].empty()) {
        int_node->output = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(
            fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicVar>(n.extra_syms[i]));
      } else {
        int_node->output = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(n.extra_ints[i]);
      }
      node->inputs.push_back(int_node);
    }
  } else if (n.axis != std::numeric_limits<int64_t>::min()) {
    auto axis_node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
    axis_node->op = fxrt::ops::Op_End;
    // A non-empty sym_axis means one non-tensor input is a symbolic scalar (a Dynamo-lifted
    // SymInt placeholder); rebuild it as a Symbol Value, the same channel a symbolic
    // shape-list element takes, so the lowering can refresh the materialized payload per
    // Execute. The axis is the main case; the channel carries any single symbolic scalar.
    if (!n.sym_axis.empty()) {
      axis_node->output =
          fxrt::ir::MakeIntrusive<fxrt::ir::Value>(fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicVar>(n.sym_axis));
    } else {
      axis_node->output = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(n.axis);
    }
    node->inputs.push_back(axis_node);
  }
  if (n.has_scalar) {
    auto scalar_node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
    scalar_node->op = fxrt::ops::Op_End;
    scalar_node->output = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(n.scalar);
    node->inputs.push_back(scalar_node);
  }
  // Shape-list (Expand / BroadcastTo / TopKV2 k): rebuild as Tuple<int>; Path C materializes
  // a shape-tensor dual view (Launch payload + tiling StorageShape).
  if (!n.int_list.empty()) {
    std::vector<fxrt::ir::ValuePtr> elems;
    elems.reserve(n.int_list.size());
    for (size_t i = 0; i < n.int_list.size(); ++i) {
      // A slot with a symbol text is a dynamic dim: rebuild it as a Symbol Value so the
      // lowering keeps the expression instead of a placeholder integer.
      if (i < n.sym_list.size() && !n.sym_list[i].empty()) {
        elems.push_back(
          fxrt::ir::MakeIntrusive<fxrt::ir::Value>(fxrt::ir::MakeIntrusive<fxrt::ir::SymbolicVar>(n.sym_list[i])));
        continue;
      }
      elems.push_back(fxrt::ir::MakeIntrusive<fxrt::ir::Value>(n.int_list[i]));
    }
    auto list_node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
    list_node->op = fxrt::ops::Op_End;
    list_node->output =
        fxrt::ir::MakeIntrusive<fxrt::ir::Value>(fxrt::ir::MakeIntrusive<fxrt::ir::Tuple>(std::move(elems)));
    node->inputs.push_back(list_node);
  }
  // Bool attrs (TopKV2 sorted/largest in GE order).
  for (int64_t b : n.bool_list) {
    auto bool_node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
    bool_node->op = fxrt::ops::Op_End;
    bool_node->output = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(b != 0);
    node->inputs.push_back(bool_node);
  }
}

static void AddComputeOpNode(const IrNodeSpec &n, const fxrt::ir::GraphPtr &graph,
                            std::unordered_map<std::string, fxrt::ir::NodePtr> &by_name) {
  // Compute op: must match ops.list exactly. Unknown / custom_call names stay as
  // Op_custom_call and Convert rejects them (no ATen alias rewriting).
  auto node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
  fxrt::ops::Op matched = fxrt::ops::Op_End;
  std::string fallback_name;
  if (TryMatchOp(n.op, &matched) && matched != fxrt::ops::Op_custom_call) {
    node->op = matched;
  } else {
    node->op = fxrt::ops::Op_custom_call;
    auto name_node = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
    name_node->op = fxrt::ops::Op_End;
    fallback_name = n.op;
    if (!n.overload.empty()) {
      fallback_name += "." + n.overload;
    }
    name_node->output = fxrt::ir::MakeIntrusive<fxrt::ir::Value>(std::string(fallback_name));
    node->inputs.push_back(name_node);
  }
  if (PreferOrderedArguments(node->op, n)) {
    // The exporter repeats the qualified custom_call name as the first ordered
    // argument; the synthesized name node already carries it.
    const size_t arg_begin =
        (node->op == fxrt::ops::Op_custom_call && !n.arguments.empty() &&
         n.arguments[0].kind == IrValueSpec::Kind::kString && n.arguments[0].string_value == fallback_name)
            ? 1U
            : 0U;
    AddOrderedArguments(n, graph, by_name, node, arg_begin);
  } else {
    // Export schema 5 retains rank-0 as_strided's empty size/stride lists in
    // ordered arguments, but its legacy int_list channel omits them. Recover
    // the unambiguous scalar geometry without changing the torch_npu ABI.
    if (node->op == fxrt::ops::Op_as_strided_view && n.int_list.empty() && n.arguments.size() >= 3U &&
        n.arguments[1].kind == IrValueSpec::Kind::kTuple && n.arguments[1].elements.empty() &&
        n.arguments[2].kind == IrValueSpec::Kind::kTuple && n.arguments[2].elements.empty()) {
      auto scalar = n;
      if (n.arguments.size() > 3U && n.arguments[3].kind != IrValueSpec::Kind::kInt &&
          n.arguments[3].kind != IrValueSpec::Kind::kNone) {
        throw std::invalid_argument("rank-0 as_strided requires a concrete storage offset");
      }
      scalar.int_list = {0, n.arguments.size() > 3U ? n.arguments[3].int_value : 0};
      AddNormalizedInputs(scalar, by_name, node);
    } else {
      AddNormalizedInputs(n, by_name, node);
    }
  }
  // Multi-output (TopKV2 / fallback Tensor[]): rebuild Tuple so tuple_getitem can index them.
  const bool fallback_op = node->op == fxrt::ops::Op_custom_call || node->op == fxrt::ops::Op_python_call;
  const size_t min_outs = fallback_op ? 1U : 2U;
  if (n.output_shapes.size() >= min_outs) {
    std::vector<fxrt::ir::ValuePtr> outs;
    outs.reserve(n.output_shapes.size());
    for (size_t i = 0; i < n.output_shapes.size(); ++i) {
      const std::string dt = (i < n.output_dtypes.size()) ? n.output_dtypes[i] : n.dtype;
      // Per-slot symbolic shapes when the export carries them (LayerNormV4); slot 0 falls
      // back to the node-level sym_shape for older exports (TopKV2).
      const std::vector<std::string> sym = (i < n.output_sym_shapes.size()) ? n.output_sym_shapes[i]
                                           : (i == 0)                         ? n.sym_shape
                                                                              : std::vector<std::string>{};
      auto slot = MakeMetaTensorValue(n.output_shapes[i], sym, dt);
      const int32_t slot_format = (i < n.output_formats.size()) ? n.output_formats[i] : n.format;
      ApplyFormatCode(slot, slot_format);
      outs.push_back(std::move(slot));
    }
    node->output =
        fxrt::ir::MakeIntrusive<fxrt::ir::Value>(fxrt::ir::MakeIntrusive<fxrt::ir::Tuple>(std::move(outs)));
  } else {
    node->output = MakeMetaTensorValue(n.shape, n.sym_shape, n.dtype);
    ApplyTensorLayout(n, node->output);
  }
  for (const auto &attr : n.attrs) {
    const auto attr_node = MakeArgumentNode(attr.second, graph, by_name);
    if (attr_node != nullptr) {
      node->attrs[attr.first] = attr_node->output;
    }
  }
  graph->nodes.push_back(node);
  by_name[n.name] = node;
}

fxrt::ir::GraphPtr BuildMrtIrGraphFromSpec(const IrGraphSpec &spec) {
  auto graph = fxrt::ir::MakeIntrusive<fxrt::ir::Graph>();
  std::unordered_map<std::string, fxrt::ir::NodePtr> by_name;

  const std::unordered_set<std::string> real_refs = CollectRealOpInputRefs(spec);
  for (const auto &n : spec.nodes) {
    if (n.kind == IrNodeKind::kInput || n.kind == IrNodeKind::kParameter) {
      AddInputOrParameterNode(n, graph, by_name, real_refs);
    } else if (n.kind == IrNodeKind::kReturn) {
      AddReturnNode(n, graph, by_name);
    } else {
      AddComputeOpNode(n, graph, by_name);
    }
  }

  // Some export paths omit an explicit return; treat the last compute op as the graph output.
  bool has_return = std::any_of(
      graph->nodes.begin(), graph->nodes.end(),
      [](const auto &n) { return n != nullptr && n->op == fxrt::ops::Op_return; });
  if (!has_return) {
    for (auto it = graph->nodes.rbegin(); it != graph->nodes.rend(); ++it) {
      const auto &n = *it;
      if (n == nullptr || n->op == fxrt::ops::Op_End || n->op == fxrt::ops::Op_return) {
        continue;
      }
      if (n->output == nullptr || !n->output->IsTensor()) {
        continue;
      }
      auto ret = fxrt::ir::MakeIntrusive<fxrt::ir::Node>();
      ret->op = fxrt::ops::Op_return;
      ret->inputs.push_back(n);
      graph->nodes.push_back(ret);
      break;
    }
  }
  return graph;
}

}  // namespace fx_converter
