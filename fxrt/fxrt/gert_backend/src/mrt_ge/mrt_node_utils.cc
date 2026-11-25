#include "mrt_ge/mrt_node_utils.h"

#include <algorithm>

#include "common/logger.h"
#include "ir/tensor/tensor.h"
#include "ir/symbolic/symbolic.h"
#include "ir/value/value.h"
#include "ops/op_def/ops_name.h"

namespace fx_converter {

std::vector<int64_t> ShapeOfTensor(const fxrt::ir::TensorPtr &tensor) {
  if (tensor == nullptr) {
    return {-1, -1};
  }
  if (tensor->HasSymbolicShape() && !tensor->GetSymbolicShape().empty()) {
    const auto &sym = tensor->GetSymbolicShape();
    std::vector<int64_t> out(sym.size());
    for (size_t i = 0; i < sym.size(); ++i) {
      if (auto c = dynamic_cast<const fxrt::ir::SymbolicConst *>(sym[i].get())) {
        out[i] = c->GetValue();
      } else {
        out[i] = -1;
      }
    }
    return out;
  }
  if (tensor->Shape().empty()) {
    return {};
  }
  std::vector<int64_t> out = tensor->Shape();
  std::replace_if(out.begin(), out.end(), [](int64_t d) { return d <= 0; }, -1);
  return out;
}

std::vector<int64_t> ShapeOfNode(const fxrt::ir::NodePtr &node) {
  if (node == nullptr || node->output == nullptr || !node->output->IsTensor()) {
    return {-1, -1};
  }
  return ShapeOfTensor(node->output->ToTensor());
}

ge::DataType MrtToGeDataType(fxrt::ir::DataType dtype, bool *mapped) {
  if (mapped != nullptr) {
    *mapped = true;
  }
  switch (dtype.value) {
    case fxrt::ir::DataType::Float32:
      return ge::DT_FLOAT;
    case fxrt::ir::DataType::Float16:
      return ge::DT_FLOAT16;
    case fxrt::ir::DataType::BFloat16:
      return ge::DT_BF16;
    case fxrt::ir::DataType::Int8:
      return ge::DT_INT8;
    case fxrt::ir::DataType::Int16:
      return ge::DT_INT16;
    case fxrt::ir::DataType::Int32:
      return ge::DT_INT32;
    case fxrt::ir::DataType::Int64:
      return ge::DT_INT64;
    case fxrt::ir::DataType::UInt8:
      return ge::DT_UINT8;
    case fxrt::ir::DataType::Bool:
      return ge::DT_BOOL;
    case fxrt::ir::DataType::Float64:
      return ge::DT_DOUBLE;
    default:
      if (mapped != nullptr) {
        *mapped = false;
      }
      RT_GLOG(WARNING) << "FxConverter: unsupported MRT dtype=" << dtype.ToString()
                       << "; falling back to DT_FLOAT";
      // TODO(jiaorui): reject Convert on unsupported dtype instead of silent Float32 fallback.
      return ge::DT_FLOAT;
  }
}

ge::DataType DtypeOfTensor(const fxrt::ir::TensorPtr &tensor) {
  if (tensor == nullptr) {
    // TODO(jiaorui): require IR dtype on every tensor edge; fail Convert when missing.
    return ge::DT_FLOAT;
  }
  bool mapped = false;
  const ge::DataType ge_dt = MrtToGeDataType(tensor->Dtype(), &mapped);
  if (!mapped) {
    RT_GLOG(WARNING) << "FxConverter: unsupported MRT dtype=" << tensor->Dtype().ToString()
                     << "; falling back to DT_FLOAT";
  }
  return ge_dt;
}

ge::DataType DtypeOfNode(const fxrt::ir::NodePtr &node) {
  if (node == nullptr || node->output == nullptr || !node->output->IsTensor()) {
    return ge::DT_FLOAT;
  }
  return DtypeOfTensor(node->output->ToTensor());
}

std::vector<fxrt::ir::TensorPtr> TensorsOfNode(const fxrt::ir::NodePtr &node) {
  std::vector<fxrt::ir::TensorPtr> slots;
  if (node == nullptr || node->output == nullptr) {
    return slots;
  }
  if (node->output->IsTensor()) {
    slots.push_back(node->output->ToTensor());
    return slots;
  }
  if (!node->output->IsTuple()) {
    return slots;
  }
  const auto tup = node->output->ToTuple();
  if (tup == nullptr) {
    return slots;
  }
  slots.reserve(tup->Size());
  for (size_t i = 0; i < tup->Size(); ++i) {
    const auto &el = (*tup)[i];
    slots.push_back((el != nullptr && el->IsTensor()) ? el->ToTensor() : nullptr);
  }
  return slots;
}

// The aten.item fold arrives as a custom_call whose first input is the "aten.item"
// string literal; a rank-0 placeholder consumed only that way is a Dynamo-lifted
// scalar the import folds into a constant, not a real tensor input.
bool IsItemFoldConsumer(const fxrt::ir::NodePtr &node) {
  if (node == nullptr || node->op != fxrt::ops::Op_custom_call || node->inputs.empty()) {
    return false;
  }
  const auto &callee = node->inputs[0];
  return callee != nullptr && callee->output != nullptr && callee->output->IsString() &&
         callee->output->ToString() == "aten.item";
}

// A rank-0 placeholder consumed by a compute node as a tensor input is a real feed
// (e.g. the rank-0 scales of npu_quantize). Consumption, not shape, separates it from
// a Dynamo-lifted scalar (whose only consumer is the item fold).
bool ConsumedAsTensorInput(const fxrt::ir::Graph &graph, const fxrt::ir::Node *feed) {
  if (feed == nullptr) {
    return false;
  }
  for (const auto &node : graph.nodes) {
    if (node == nullptr || node->op == fxrt::ops::Op_End || IsItemFoldConsumer(node)) {
      continue;
    }
    const auto &ins = node->inputs;
    if (std::any_of(ins.begin(), ins.end(), [feed](const fxrt::ir::NodePtr &in) { return in.get() == feed; })) {
      return true;
    }
  }
  return false;
}

bool IsModelFeedTensor(const fxrt::ir::Graph &graph, const fxrt::ir::NodePtr &node) {
  if (node == nullptr || node->output == nullptr || !node->output->IsTensor()) {
    return false;
  }
  const auto t = node->output->ToTensor();
  if (t == nullptr) {
    return false;
  }
  // Empty shape without symbolic dims: a Dynamo-lifted scalar (filtered) or a real
  // rank-0 op input (kept) -- decide by how the graph consumes it.
  if (t->Shape().empty() && !t->HasSymbolicShape()) {
    return ConsumedAsTensorInput(graph, node.get());
  }
  return true;
}

}  // namespace fx_converter
