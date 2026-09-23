/**
 * @file ir_graph_export.cpp
 * @brief Implementation of IR graph export utilities.
 *
 * TEMPORARY SOLUTION: Implementation of graph serialization functions.
 * Should be removed once ABI consistency is properly resolved.
 *
 * Original location: runtime/executor/executor.cpp (lines 322-617)
 * Commits: 496e33e (fx converter), 8c130fa (expand)
 */

#include "runtime/executor/ir_graph_export.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ir/graph.h"
#include "ops/operator.h"

namespace fxrt {
namespace runtime {

// Anonymous namespace for internal helpers
namespace {

/**
 * @brief Extract shape from value for export.
 * @param value IR value pointer
 * @return Shape as vector of int64_t (-1 for dynamic dims)
 */
std::vector<int64_t> ShapeForExport(const ir::ValuePtr& value) {
  std::vector<int64_t> dims;
  if (value == nullptr || !value->IsTensor()) {
    return dims;
  }
  const auto& tensor = value->ToTensor();
  if (tensor == nullptr) {
    return dims;
  }
  const auto& shape = tensor->Shape();
  dims.reserve(shape.size());
  if (tensor->HasDynamicShape() || tensor->HasSymbolicShape()) {
    for (size_t i = 0; i < shape.size(); ++i) {
      dims.push_back((shape[i] >= 0) ? shape[i] : -1);
    }
    return dims;
  }
  return shape;
}

/**
 * @brief Extract symbolic shape expressions from value.
 * @param value IR value pointer
 * @return Symbolic shape expressions as strings
 */
std::vector<std::string> SymShapeForExport(const ir::ValuePtr& value) {
  std::vector<std::string> dims;
  if (value == nullptr || !value->IsTensor()) {
    return dims;
  }
  const auto& tensor = value->ToTensor();
  if (tensor == nullptr) {
    return dims;
  }
  if (tensor->HasSymbolicShape()) {
    const auto& sym_shape = tensor->GetSymbolicShape();
    std::transform(sym_shape.begin(), sym_shape.end(), std::back_inserter(dims), [](const auto& expr) {
      if (expr == nullptr) {
        return std::string("");
      }
      if (expr->GetKind() == fxrt::ir::SymbolicExpr::Kind::Constant) {
        return "c:" + expr->ToString();
      }
      return expr->ToString();
    });
    if (!dims.empty()) {
      return dims;
    }
  }
  // Static graphs: emit concrete dim strings so converter InferShape still gets a rule
  // when SymbolEnv has 0 free-var seeds (as_strided / empty_strided).
  for (int64_t d : ShapeForExport(value)) {
    if (d >= 0) {
      dims.push_back(std::to_string(d));
    }
  }
  return dims;
}

/**
 * @brief Get data type string from value.
 * @param value IR value pointer
 * @return Data type as string (default: "float32")
 */
std::string DtypeOfValue(const ir::ValuePtr& value) {
  if (value == nullptr || !value->IsTensor()) {
    return "float32";
  }
  const auto& tensor = value->ToTensor();
  if (tensor == nullptr) {
    return "float32";
  }
  try {
    return tensor->Dtype().ToString();
  } catch (...) {
    return "float32";
  }
}

/**
 * @brief Check if node represents a model feed (input/parameter).
 * @param node IR node pointer
 * @return true if node is a valid model feed
 */
bool IsModelFeed(const ir::NodePtr& node) {
  if (node == nullptr || node->output == nullptr || !node->output->IsTensor()) {
    return false;
  }
  const auto& tensor = node->output->ToTensor();
  if (tensor == nullptr) {
    return false;
  }
  if (tensor->Shape().empty() && !tensor->HasSymbolicShape()) {
    return false;
  }
  return true;
}

/**
 * @brief Helper to push export node to graph.
 */
void PushExportNode(
    IrGraphExport& out,
    const std::string& name,
    const std::string& kind,
    const std::string& op,
    const std::vector<std::string>& inputs,
    const std::vector<int64_t>& shape,
    const std::vector<std::string>& sym_shape,
    const std::string& dtype = "float32",
    int64_t axis = std::numeric_limits<int64_t>::min(),
    bool has_scalar = false,
    double scalar = 0.0,
    const std::vector<int64_t>& int_list = {},
    const std::vector<int64_t>& bool_list = {},
    const std::vector<std::vector<int64_t>>& output_shapes = {},
    const std::vector<std::string>& output_dtypes = {},
    const std::vector<IrGraphNodeExport::ValueExport>& arguments = {},
    const std::unordered_map<std::string, IrGraphNodeExport::ValueExport>& attrs = {}) {
  IrGraphNodeExport n;
  n.name = name;
  n.kind = kind;
  n.op = op;
  n.inputs = inputs;
  n.shape = shape;
  n.sym_shape = sym_shape;
  n.dtype = dtype;
  n.axis = axis;
  n.has_scalar = has_scalar;
  n.scalar = scalar;
  n.int_list = int_list;
  n.bool_list = bool_list;
  n.output_shapes = output_shapes;
  n.output_dtypes = output_dtypes;
  n.arguments = arguments;
  n.attrs = attrs;
  out.nodes.push_back(std::move(n));
}

using ValueExport = IrGraphNodeExport::ValueExport;

ValueExport ExportValue(const ir::ValuePtr& value) {
  ValueExport result;
  if (value == nullptr || value->IsNone()) {
    return result;
  }
  if (value->IsInt()) {
    result.kind = ValueExport::Kind::kInt;
    result.int_value = value->ToInt();
  } else if (value->IsDouble()) {
    result.kind = ValueExport::Kind::kDouble;
    result.double_value = value->ToDouble();
  } else if (value->IsBool()) {
    result.kind = ValueExport::Kind::kBool;
    result.bool_value = value->ToBool();
  } else if (value->IsString()) {
    result.kind = ValueExport::Kind::kString;
    result.string_value = value->ToString();
  } else if (value->IsSymbol()) {
    result.kind = ValueExport::Kind::kSymbol;
    const auto symbol = value->ToSymbol();
    result.string_value = symbol == nullptr ? "" : symbol->ToString();
  } else if (value->IsTuple()) {
    result.kind = ValueExport::Kind::kTuple;
    const auto tuple = value->ToTuple();
    if (tuple != nullptr) {
      for (const auto& element : *tuple) {
        result.elements.push_back(ExportValue(element));
      }
    }
  }
  return result;
}

ValueExport ExportNodeValue(const ir::NodePtr& node, const std::unordered_map<const ir::Node*, std::string>& name_of) {
  if (node == nullptr) {
    return {};
  }
  const auto named = name_of.find(node.get());
  if (named != name_of.end()) {
    ValueExport result;
    result.kind = ValueExport::Kind::kRef;
    result.ref = named->second;
    return result;
  }
  if (node->op == ops::Op_make_tuple) {
    ValueExport result;
    result.kind = ValueExport::Kind::kTuple;
    for (const auto& element : node->inputs) {
      result.elements.push_back(ExportNodeValue(element, name_of));
    }
    return result;
  }
  return ExportValue(node->output);
}

std::vector<ValueExport> ExportArguments(
    const ir::NodePtr& node,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  std::vector<ValueExport> result;
  const auto& arguments = node->inputs;
  result.reserve(arguments.size());
  std::transform(
      arguments.begin(), arguments.end(), std::back_inserter(result), [&name_of](const ir::NodePtr& argument) {
        return ExportNodeValue(argument, name_of);
      });
  return result;
}

std::unordered_map<std::string, ValueExport> ExportAttrs(const ir::NodePtr& node) {
  std::unordered_map<std::string, ValueExport> result;
  if (node == nullptr) {
    return result;
  }
  for (const auto& attr : node->attrs) {
    result.emplace(attr.first, ExportValue(attr.second));
  }
  return result;
}

/**
 * @brief Resolve one return argument name, unwrapping a trivial make_tuple of one tensor.
 */
std::string ResolveOneReturnArg(
    const ir::NodePtr& root,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  if (root == nullptr) {
    return "";
  }
  auto it = name_of.find(root.get());
  if (it != name_of.end()) {
    return it->second;
  }
  if (root->op == ops::Op_make_tuple && root->inputs.size() == 1) {
    return ResolveOneReturnArg(root->inputs[0], name_of);
  }
  return "";
}

/**
 * @brief Resolve graph return tensor names (GE NetOutput N inputs).
 * Unwraps make_tuple of N tensor producers (e.g. topk values+indices).
 */
std::vector<std::string> ResolveReturnArgs(
    const ir::NodePtr& root,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  std::vector<std::string> out;
  if (root == nullptr) {
    return out;
  }
  if (root->op == ops::Op_make_tuple) {
    out.reserve(root->inputs.size());
    for (const auto& in : root->inputs) {
      const std::string name = ResolveOneReturnArg(in, name_of);
      if (!name.empty()) {
        out.push_back(name);
      }
    }
    return out;
  }
  const std::string name = ResolveOneReturnArg(root, name_of);
  if (!name.empty()) {
    out.push_back(name);
  }
  return out;
}

/**
 * @brief Export graph inputs.
 */
void ExportIrInputs(
    const ir::Graph* graph,
    IrGraphExport* out,
    std::unordered_map<const ir::Node*, std::string>* name_of) {
  size_t export_input_idx = 0;
  for (size_t i = 0; i < graph->inputs.size(); ++i) {
    const auto& node = graph->inputs[i];
    if (!IsModelFeed(node)) {
      continue;
    }
    const std::string name = "input_" + std::to_string(export_input_idx++);
    (*name_of)[node.get()] = name;
    PushExportNode(
        *out,
        name,
        "input",
        "",
        {},
        ShapeForExport(node->output),
        SymShapeForExport(node->output),
        DtypeOfValue(node->output));
  }
}

/**
 * @brief Export graph parameters.
 */
void ExportIrParameters(
    const ir::Graph* graph,
    IrGraphExport* out,
    std::unordered_map<const ir::Node*, std::string>* name_of) {
  size_t export_param_idx = 0;
  for (size_t i = 0; i < graph->parameters.size(); ++i) {
    const auto& node = graph->parameters[i];
    if (!IsModelFeed(node)) {
      continue;
    }
    const std::string name = "param_" + std::to_string(export_param_idx++);
    (*name_of)[node.get()] = name;
    PushExportNode(
        *out,
        name,
        "parameter",
        "",
        {},
        ShapeForExport(node->output),
        SymShapeForExport(node->output),
        DtypeOfValue(node->output));
  }
}

/**
 * @brief Helper struct for operation name resolution.
 */
struct ExportOpName {
  std::string name;
  size_t arg_begin = 0;
};

/**
 * @brief Resolve operation name for export.
 */
ExportOpName ResolveExportOpName(const ir::NodePtr& node) {
  ExportOpName result{ops::ToStr(node->op), 0};
  if (node->op == ops::Op_custom_call && !node->inputs.empty()) {
    const auto& name_node = node->inputs[0];
    if (name_node != nullptr && name_node->output != nullptr && name_node->output->IsString()) {
      result.name = name_node->output->ToString();
      result.arg_begin = 1;
    }
  }
  return result;
}

/**
 * @brief Collect make_tuple inputs.
 */
void CollectMakeTupleInputs(
    const ir::NodePtr& tuple_node,
    const std::unordered_map<const ir::Node*, std::string>& name_of,
    std::vector<std::string>* inputs) {
  for (const auto& t : tuple_node->inputs) {
    if (t == nullptr) {
      continue;
    }
    auto it = name_of.find(t.get());
    inputs->push_back(it != name_of.end() ? it->second : ("unknown_" + std::to_string(inputs->size())));
  }
}

/**
 * @brief Try to capture scalar value.
 */
bool TryCaptureScalar(const ir::ValuePtr& value, bool* has_scalar, double* scalar) {
  if (value == nullptr) {
    return false;
  }
  if (value->IsDouble()) {
    *has_scalar = true;
    *scalar = value->ToDouble();
    return true;
  }
  if (value->IsBool()) {
    *has_scalar = true;
    *scalar = value->ToBool() ? 1.0 : 0.0;
    return true;
  }
  return false;
}

/**
 * @brief Helper struct for collecting export inputs.
 */
struct ExportInputs {
  std::vector<std::string> names;
  int64_t axis = std::numeric_limits<int64_t>::min();
  bool has_scalar = false;
  double scalar = 0.0;
  std::vector<int64_t> int_list;
  std::vector<int64_t> bool_list;
};

/**
 * @brief Try to capture int list from tuple.
 * True when make_tuple is an int-size list (expand / BroadcastTo shape), not tensor packing (cat).
 */
bool TryCaptureIntListTuple(const ir::NodePtr& tuple_node, std::vector<int64_t>* int_list) {
  if (tuple_node == nullptr || tuple_node->inputs.empty() || int_list == nullptr) {
    return false;
  }
  std::vector<int64_t> dims;
  dims.reserve(tuple_node->inputs.size());
  for (const auto& t : tuple_node->inputs) {
    if (t == nullptr || t->output == nullptr || !t->output->IsInt()) {
      return false;
    }
    dims.push_back(t->output->ToInt());
  }
  *int_list = std::move(dims);
  return true;
}

/**
 * @brief Collect constant_pad_nd: pad int-list + pad value scalar (Int or float).
 * Generic Collect would put Int value into axis; specialized path keeps has_scalar.
 * Pad list stays PyTorch last-dim-first; Path C converts to GE PadV3 contiguous.
 */
ExportInputs CollectConstantPadExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  ExportInputs result;
  for (size_t i = arg_begin; i < node->inputs.size(); ++i) {
    const auto& in = node->inputs[i];
    if (in == nullptr || in->output == nullptr) {
      continue;
    }
    auto it = name_of.find(in.get());
    if (it != name_of.end()) {
      result.names.push_back(it->second);
      continue;
    }
    if (in->op == ops::Op_make_tuple) {
      if (TryCaptureIntListTuple(in, &result.int_list)) {
        continue;
      }
      CollectMakeTupleInputs(in, name_of, &result.names);
      continue;
    }
    if (in->output->IsTuple()) {
      // Unnamed Tuple Value (rare); flatten int elems when possible.
      const auto& tup = in->output->ToTuple();
      if (tup != nullptr) {
        std::vector<int64_t> dims = tup->ToIntList();
        if (!dims.empty()) {
          result.int_list = std::move(dims);
          continue;
        }
      }
    }
    if (in->output->IsDouble()) {
      result.has_scalar = true;
      result.scalar = in->output->ToDouble();
      continue;
    }
    if (in->output->IsInt()) {
      // Pad value (not axis). Prefer first Int as value when pad list already captured.
      if (!result.has_scalar) {
        result.has_scalar = true;
        result.scalar = static_cast<double>(in->output->ToInt());
      }
      continue;
    }
    if (in->output->IsBool()) {
      continue;
    }
    if (in->output->IsTensor()) {
      result.names.push_back("unknown_" + std::to_string(i));
    }
  }
  // GE PadV3 constant_values defaults to 0 when omitted.
  if (!result.has_scalar) {
    result.has_scalar = true;
    result.scalar = 0.0;
  }
  return result;
}

/**
 * @brief Collect full_like / mul_scalar fill/mul value as has_scalar (Int or float; not axis).
 */
ExportInputs CollectFullLikeExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  ExportInputs result;
  for (size_t i = arg_begin; i < node->inputs.size(); ++i) {
    const auto& in = node->inputs[i];
    if (in == nullptr || in->output == nullptr) {
      continue;
    }
    auto it = name_of.find(in.get());
    if (it != name_of.end()) {
      result.names.push_back(it->second);
      continue;
    }
    if (in->output->IsTensor()) {
      result.names.push_back("unknown_" + std::to_string(i));
      continue;
    }
    if (in->output->IsDouble()) {
      result.has_scalar = true;
      result.scalar = in->output->ToDouble();
      continue;
    }
    if (in->output->IsInt()) {
      if (!result.has_scalar) {
        result.has_scalar = true;
        result.scalar = static_cast<double>(in->output->ToInt());
      }
      continue;
    }
    // Skip dtype / layout / device / pin_memory / memory_format kwargs.
  }
  return result;
}

/**
 * @brief Collect reduce_sum: dims → int_list, keepdim → bool_list[0]; skip dtype Int.
 */
ExportInputs CollectReduceSumExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  ExportInputs result;
  bool keep_dims = false;
  bool saw_keep_dims = false;
  for (size_t i = arg_begin; i < node->inputs.size(); ++i) {
    const auto& in = node->inputs[i];
    if (in == nullptr || in->output == nullptr) {
      continue;
    }
    auto it = name_of.find(in.get());
    if (it != name_of.end()) {
      result.names.push_back(it->second);
      continue;
    }
    if (in->op == ops::Op_make_tuple) {
      if (TryCaptureIntListTuple(in, &result.int_list)) {
        continue;
      }
      continue;
    }
    if (in->output->IsTensor()) {
      result.names.push_back("unknown_" + std::to_string(i));
      continue;
    }
    if (in->output->IsBool()) {
      keep_dims = in->output->ToBool();
      saw_keep_dims = true;
      continue;
    }
    // Skip dtype (Int / other) — GE ReduceSum has no dtype attr.
  }
  if (saw_keep_dims) {
    result.bool_list = {keep_dims ? 1 : 0};
  }
  return result;
}

/**
 * @brief Collect TopKV2 attrs: IR order is k, dim, largest, sorted.
 * Export: int_list=[k], axis=dim, bool_list=[sorted, largest] (GE attr order).
 */
ExportInputs CollectTopkExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  ExportInputs result;
  std::vector<int64_t> ints;
  std::vector<bool> bools;
  for (size_t i = arg_begin; i < node->inputs.size(); ++i) {
    const auto& in = node->inputs[i];
    if (in == nullptr || in->output == nullptr) {
      continue;
    }
    auto it = name_of.find(in.get());
    if (it != name_of.end()) {
      result.names.push_back(it->second);
      continue;
    }
    if (in->output->IsTensor()) {
      result.names.push_back("unknown_" + std::to_string(i));
      continue;
    }
    if (in->output->IsInt()) {
      ints.push_back(in->output->ToInt());
      continue;
    }
    if (in->output->IsBool()) {
      bools.push_back(in->output->ToBool());
      continue;
    }
  }
  // k (required), dim (default -1)
  if (!ints.empty()) {
    result.int_list = {ints[0]};
  }
  if (ints.size() >= 2) {
    result.axis = ints[1];
  } else {
    result.axis = -1;
  }
  // IR bools: largest, sorted → GE bool_list: sorted, largest
  bool largest = true;
  bool sorted = true;
  if (!bools.empty()) {
    largest = bools[0];
  }
  if (bools.size() >= 2) {
    sorted = bools[1];
  }
  result.bool_list = {sorted ? 1 : 0, largest ? 1 : 0};
  return result;
}

/**
 * @brief Collect inputs for export.
 */
/**
 * @brief Collect slice_view / narrow_view Int args.
 * aten.slice.Tensor(self, dim, start, end, step=1) → axis=dim, int_list=[start,end,step].
 * aten.narrow(self, dim, start, length) → axis=dim, int_list=[start,start+length,1].
 * Generic Collect would overwrite axis with each Int and drop start/end/step.
 */
ExportInputs CollectSliceViewExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::string& op_name,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  ExportInputs result;
  std::vector<int64_t> ints;
  for (size_t i = arg_begin; i < node->inputs.size(); ++i) {
    const auto& in = node->inputs[i];
    if (in == nullptr || in->output == nullptr) {
      continue;
    }
    auto it = name_of.find(in.get());
    if (it != name_of.end()) {
      result.names.push_back(it->second);
      continue;
    }
    if (in->output->IsInt()) {
      ints.push_back(in->output->ToInt());
      continue;
    }
  }
  const bool is_narrow = (op_name == "narrow_view" || op_name == "narrow");
  // slice: dim, start, end [, step]; narrow: dim, start, length
  if (ints.size() >= 3U) {
    result.axis = ints[0];
    const int64_t start = ints[1];
    if (is_narrow) {
      const int64_t length = ints[2];
      result.int_list = {start, start + length, 1};
    } else {
      const int64_t step = (ints.size() >= 4U) ? ints[3] : 1;
      result.int_list = {start, ints[2], step};
    }
  } else if (ints.size() == 1U) {
    result.axis = ints[0];
  }
  return result;
}

/**
 * @brief Collect getitem_slice: begin/end/axes/steps int tuples.
 * Pack int_list as [n, start0,end0,step0,axis0, ..., start{n-1},end{n-1},step{n-1},axis{n-1}].
 * Path C expands to full-rank Slice offsets+size (step==1 only).
 */
/**
 * @brief Collect select_view / select: dim + index Int args.
 * aten.select.int(self, dim, index) → axis=dim, int_list=[index] (Path C GatherV2).
 */
ExportInputs CollectSelectViewExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  ExportInputs result;
  std::vector<int64_t> ints;
  for (size_t i = arg_begin; i < node->inputs.size(); ++i) {
    const auto& in = node->inputs[i];
    if (in == nullptr || in->output == nullptr) {
      continue;
    }
    auto it = name_of.find(in.get());
    if (it != name_of.end()) {
      result.names.push_back(it->second);
      continue;
    }
    if (in->output->IsInt()) {
      ints.push_back(in->output->ToInt());
      continue;
    }
  }
  // Expected: dim, index.
  if (ints.size() >= 2U) {
    result.axis = ints[0];
    result.int_list = {ints[1]};
  } else if (ints.size() == 1U) {
    result.axis = ints[0];
  }
  return result;
}

ExportInputs CollectGetitemSliceExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  ExportInputs result;
  std::vector<std::vector<int64_t>> lists;
  lists.reserve(4U);
  for (size_t i = arg_begin; i < node->inputs.size(); ++i) {
    const auto& in = node->inputs[i];
    if (in == nullptr || in->output == nullptr) {
      continue;
    }
    auto it = name_of.find(in.get());
    if (it != name_of.end()) {
      result.names.push_back(it->second);
      continue;
    }
    if (in->op == ops::Op_make_tuple) {
      std::vector<int64_t> dims;
      if (TryCaptureIntListTuple(in, &dims)) {
        lists.push_back(std::move(dims));
        continue;
      }
    }
  }
  // Expected: begin, end, axes, steps (equal length).
  if (lists.size() != 4U || lists[0].empty() || lists[0].size() != lists[1].size() ||
      lists[0].size() != lists[2].size() || lists[0].size() != lists[3].size()) {
    return result;
  }
  const size_t n = lists[0].size();
  result.int_list.clear();
  result.int_list.reserve(1U + n * 4U);
  result.int_list.push_back(static_cast<int64_t>(n));
  for (size_t i = 0; i < n; ++i) {
    result.int_list.push_back(lists[0][i]); // start
    result.int_list.push_back(lists[1][i]); // end
    result.int_list.push_back(lists[3][i]); // step
    result.int_list.push_back(lists[2][i]); // axis
  }
  return result;
}

/**
 * @brief Collect gather_v2 / index_tensor: keep tensor edges only; drop dim Int.
 * getitem_impl: gather_v2(x, 0, indices); aten.index may pack indices in make_tuple.
 * GE Gather is dim0 via attrs.
 */
ExportInputs CollectGatherV2ExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  ExportInputs result;
  for (size_t i = arg_begin; i < node->inputs.size(); ++i) {
    const auto& in = node->inputs[i];
    if (in == nullptr || in->output == nullptr) {
      continue;
    }
    auto it = name_of.find(in.get());
    if (it != name_of.end()) {
      result.names.push_back(it->second);
      continue;
    }
    // aten.index.Tensor indices list → make_tuple of tensor producers.
    if (in->op == ops::Op_make_tuple) {
      CollectMakeTupleInputs(in, name_of, &result.names);
      continue;
    }
    // Skip dim Int (always 0 for tensor getitem → Gather dim0).
    if (in->output->IsInt()) {
      continue;
    }
  }
  return result;
}

struct AsStridedCapture {
  std::vector<std::string> names;
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  bool has_size = false;
  bool has_stride = false;
  int64_t offset = 0;
  bool has_offset = false;
  size_t tuple_slot = 0;
};

static void AssignAsStridedTupleSlot(AsStridedCapture* cap, std::vector<int64_t> dims, bool captured) {
  if (cap == nullptr) {
    return;
  }
  if (cap->tuple_slot == 0) {
    if (captured) {
      cap->size = std::move(dims);
      cap->has_size = true;
    }
  } else if (cap->tuple_slot == 1 && captured) {
    cap->stride = std::move(dims);
    cap->has_stride = true;
  }
  ++cap->tuple_slot;
}

static void CaptureAsStridedArgInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::unordered_map<const ir::Node*, std::string>& name_of,
    AsStridedCapture* cap) {
  if (cap == nullptr || node == nullptr) {
    return;
  }
  for (size_t i = arg_begin; i < node->inputs.size(); ++i) {
    const auto& in = node->inputs[i];
    if (in == nullptr || in->output == nullptr) {
      continue;
    }
    auto it = name_of.find(in.get());
    if (it != name_of.end()) {
      cap->names.push_back(it->second);
      continue;
    }
    if (in->op == ops::Op_make_tuple) {
      std::vector<int64_t> dims;
      const bool captured = TryCaptureIntListTuple(in, &dims);
      AssignAsStridedTupleSlot(cap, std::move(dims), captured);
      continue;
    }
    if (in->output->IsInt()) {
      cap->offset = in->output->ToInt();
      cap->has_offset = true;
    }
  }
}

static void FallbackAsStridedSizeFromOutput(const ir::NodePtr& node, AsStridedCapture* cap) {
  if (cap == nullptr || cap->has_size || node == nullptr || node->output == nullptr) {
    return;
  }
  std::vector<int64_t> out_shape = ShapeForExport(node->output);
  if (out_shape.empty() || std::any_of(out_shape.begin(), out_shape.end(), [](int64_t d) { return d < 0; })) {
    return;
  }
  cap->size = std::move(out_shape);
  cap->has_size = true;
}

static void PackAsStridedIntList(const AsStridedCapture& cap, ExportInputs* result) {
  if (result == nullptr || !cap.has_size || !cap.has_stride || cap.size.empty() ||
      cap.size.size() != cap.stride.size()) {
    return;
  }
  const int64_t rank = static_cast<int64_t>(cap.size.size());
  result->int_list.clear();
  result->int_list.reserve(static_cast<size_t>(1 + rank + rank + 1));
  result->int_list.push_back(rank);
  result->int_list.insert(result->int_list.end(), cap.size.begin(), cap.size.end());
  result->int_list.insert(result->int_list.end(), cap.stride.begin(), cap.stride.end());
  result->int_list.push_back(cap.has_offset ? cap.offset : 0);
}

/**
 * @brief Collect as_strided_view: pack [rank]+size+stride+[offset] into int_list.
 * Contiguous+offset==0 may ViewAlias-degrade; prefix non-contig → Path C Slice.
 *
 * Size tuples often keep SymInt producers (printed as input_0 / s0) under dynamic=True;
 * TryCaptureIntListTuple then fails while stride (all Int) succeeds. Do NOT treat the
 * single captured list as size (that yielded size+offset-only packs → "strides missing").
 * Fallback: use concrete output tensor shape as size when capture fails.
 */
ExportInputs CollectAsStridedExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  AsStridedCapture cap;
  CaptureAsStridedArgInputs(node, arg_begin, name_of, &cap);
  FallbackAsStridedSizeFromOutput(node, &cap);
  ExportInputs result;
  result.names = std::move(cap.names);
  PackAsStridedIntList(cap, &result);
  return result;
}

bool TryCollectSpecialExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::string& op_name,
    const std::unordered_map<const ir::Node*, std::string>& name_of,
    ExportInputs* out) {
  if (op_name == "topk") {
    *out = CollectTopkExportInputs(node, arg_begin, name_of);
    return true;
  }
  if (op_name == "constant_pad_nd") {
    *out = CollectConstantPadExportInputs(node, arg_begin, name_of);
    return true;
  }
  if (op_name == "full_like" || op_name == "mul_scalar") {
    // Fills / Muls: scalar → has_scalar (REQUIRED_ATTR value); skip Int→axis.
    *out = CollectFullLikeExportInputs(node, arg_begin, name_of);
    return true;
  }
  if (op_name == "reduce_sum") {
    *out = CollectReduceSumExportInputs(node, arg_begin, name_of);
    return true;
  }
  if (op_name == "slice_view" || op_name == "slice" || op_name == "narrow_view" || op_name == "narrow") {
    *out = CollectSliceViewExportInputs(node, arg_begin, op_name, name_of);
    return true;
  }
  if (op_name == "select_view" || op_name == "select") {
    *out = CollectSelectViewExportInputs(node, arg_begin, name_of);
    return true;
  }
  if (op_name == "getitem_slice") {
    *out = CollectGetitemSliceExportInputs(node, arg_begin, name_of);
    return true;
  }
  if (op_name == "gather_v2" || op_name == "index_tensor") {
    *out = CollectGatherV2ExportInputs(node, arg_begin, name_of);
    return true;
  }
  if (op_name == "as_strided_view") {
    *out = CollectAsStridedExportInputs(node, arg_begin, name_of);
    return true;
  }
  return false;
}

ExportInputs CollectGenericExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  ExportInputs result;
  for (size_t i = arg_begin; i < node->inputs.size(); ++i) {
    const auto& in = node->inputs[i];
    if (in == nullptr) {
      continue;
    }
    // Named producers (including Tuple-output ops such as topk) are graph edges.
    auto named = name_of.find(in.get());
    if (named != name_of.end()) {
      result.names.push_back(named->second);
      continue;
    }
    if (in->op == ops::Op_make_tuple) {
      // GE Expand/BroadcastTo: shape is Const INT64 1D tensor; export as int_list.
      if (TryCaptureIntListTuple(in, &result.int_list)) {
        continue;
      }
      CollectMakeTupleInputs(in, name_of, &result.names);
      continue;
    }
    if (in->output != nullptr && in->output->IsInt()) {
      result.axis = in->output->ToInt();
      continue;
    }
    // Skip Bool kwargs (e.g. aten.expand implicit=False); GE Expand has only x + shape.
    if (in->output != nullptr && in->output->IsBool()) {
      continue;
    }
    if (TryCaptureScalar(in->output, &result.has_scalar, &result.scalar)) {
      continue;
    }
    if (in->output == nullptr || !in->output->IsTensor()) {
      continue;
    }
    result.names.push_back("unknown_" + std::to_string(i));
  }
  return result;
}

ExportInputs CollectExportInputs(
    const ir::NodePtr& node,
    size_t arg_begin,
    const std::string& op_name,
    const std::unordered_map<const ir::Node*, std::string>& name_of) {
  ExportInputs result;
  if (TryCollectSpecialExportInputs(node, arg_begin, op_name, name_of, &result)) {
    return result;
  }
  return CollectGenericExportInputs(node, arg_begin, name_of);
}

bool UsesOrderedArguments(const std::string& op_name) {
  return op_name == "empty_strided" || op_name == "compiled_kernel_mutation";
}

bool IsExportableComputeOutput(const ir::ValuePtr& value) {
  if (value == nullptr) {
    return false;
  }
  if (value->IsTensor()) {
    return true;
  }
  if (!value->IsTuple()) {
    return false;
  }
  const auto& tup = value->ToTuple();
  if (tup == nullptr || tup->Size() == 0) {
    return false;
  }
  for (size_t i = 0; i < tup->Size(); ++i) {
    if ((*tup)[i] == nullptr || !(*tup)[i]->IsTensor()) {
      return false;
    }
  }
  return true;
}

void FillMultiOutputMeta(
    const ir::ValuePtr& value,
    std::vector<std::vector<int64_t>>* output_shapes,
    std::vector<std::string>* output_dtypes,
    const std::string& op_name) {
  if (value == nullptr || !value->IsTuple() || output_shapes == nullptr || output_dtypes == nullptr) {
    return;
  }
  const auto& tup = value->ToTuple();
  if (tup == nullptr || tup->Size() < 2) {
    return;
  }
  output_shapes->clear();
  output_dtypes->clear();
  output_shapes->reserve(tup->Size());
  output_dtypes->reserve(tup->Size());
  for (size_t i = 0; i < tup->Size(); ++i) {
    output_shapes->push_back(ShapeForExport((*tup)[i]));
    std::string dt = DtypeOfValue((*tup)[i]);
    // GE TopKV2 indices bins are INT32 on this pack; torch meta is often int64.
    if (op_name == "topk" && i == 1) {
      dt = "int32";
    }
    output_dtypes->push_back(dt);
  }
}

/**
 * @brief Export compute operations from graph.
 */
void ExportIrComputeOps(
    const ir::Graph* graph,
    IrGraphExport* out,
    std::unordered_map<const ir::Node*, std::string>* name_of,
    std::vector<std::string>* return_args) {
  auto& name_map = *name_of;
  size_t op_idx = 0;
  for (const auto& node : graph->nodes) {
    if (node == nullptr) {
      continue;
    }
    if (node->op == ops::Op_End) {
      continue;
    }
    if (node->op == ops::Op_return) {
      if (!node->inputs.empty()) {
        *return_args = ResolveReturnArgs(node->inputs[0], name_map);
      }
      continue;
    }
    // Skip make_tuple packing; returns unwrap it. Keep tuple_getitem (tensor edges).
    if (node->op == ops::Op_make_tuple) {
      continue;
    }
    if (!IsExportableComputeOutput(node->output)) {
      continue;
    }
    const auto op_meta = ResolveExportOpName(node);
    const std::string name = "op" + std::to_string(op_idx++);
    name_map[node.get()] = name;
    const auto inputs = UsesOrderedArguments(op_meta.name)
        ? ExportInputs{}
        : CollectExportInputs(node, op_meta.arg_begin, op_meta.name, name_map);
    const auto arguments = ExportArguments(node, name_map);
    const auto attrs = ExportAttrs(node);

    std::vector<int64_t> shape;
    std::vector<std::string> sym_shape;
    std::string dtype = "float32";
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<std::string> output_dtypes;
    if (node->output->IsTensor()) {
      shape = ShapeForExport(node->output);
      sym_shape = SymShapeForExport(node->output);
      dtype = DtypeOfValue(node->output);
      // GE TopKV2 indices: force int32 when exporting getitem(1).
      if (op_meta.name == "tuple_getitem" && inputs.axis == 1) {
        dtype = "int32";
      }
    } else {
      FillMultiOutputMeta(node->output, &output_shapes, &output_dtypes, op_meta.name);
      if (!output_shapes.empty()) {
        shape = output_shapes[0];
        dtype = output_dtypes[0];
        // Primary sym_shape from first tuple element when present.
        const auto& tup = node->output->ToTuple();
        if (tup != nullptr && tup->Size() > 0) {
          sym_shape = SymShapeForExport((*tup)[0]);
        }
      }
    }
    PushExportNode(
        *out,
        name,
        "op",
        op_meta.name,
        inputs.names,
        shape,
        sym_shape,
        dtype,
        inputs.axis,
        inputs.has_scalar,
        inputs.scalar,
        inputs.int_list,
        inputs.bool_list,
        output_shapes,
        output_dtypes,
        arguments,
        attrs);
  }
}

} // anonymous namespace

/**
 * @brief Main export function to be called by GraphExecutor.
 *
 * @param graph IR graph to export
 * @param name Graph name
 * @return Exported graph structure
 */
IrGraphExport ExportIrGraph(const ir::Graph* graph, const std::string& name) {
  IrGraphExport out;
  out.name = name;
  std::unordered_map<const ir::Node*, std::string> name_of;

  ExportIrInputs(graph, &out, &name_of);
  ExportIrParameters(graph, &out, &name_of);
  std::vector<std::string> return_args;
  ExportIrComputeOps(graph, &out, &name_of, &return_args);
  if (!return_args.empty()) {
    PushExportNode(out, "ret", "return", "return", return_args, {}, {});
  }
  return out;
}

} // namespace runtime
} // namespace fxrt
