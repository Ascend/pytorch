#include "exe_graph/es_graph_builder.h"

#include <mutex>

#include "common/logger.h"
#include "exe_graph/fxrt_kernel_registration.h"
#include "ge/compliant_node_builder.h"

namespace fx_converter {
namespace {

// Keep in sync with execute_graph_metadata.h (Mix launch: stream[1] + notify[2]).
constexpr int64_t kModelStreamCountValue = 2;
constexpr int64_t kModelNotifyCountValue = 2;

constexpr const char *kConstValue = "value";
constexpr const char *kFeedIndex = "index";
constexpr const char *kModelInputs = "gert.model.inputs";
constexpr const char *kModelOutputs = "gert.model.outputs";
constexpr const char *kModelStreamCount = "gert.model.stream_count";
constexpr const char *kModelEventCount = "gert.model.event_count";
constexpr const char *kModelNotifyCount = "gert.model.notify_count";
constexpr const char *kModelAttachedStreamCount = "gert.model.attached_stream_count";

std::mutex &RegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<ge::Graph *, FxGraphBuilder *> &Registry() {
  static std::unordered_map<ge::Graph *, FxGraphBuilder *> registry;
  return registry;
}

ge::Tensor ToModelIoTensor(const ge::TensorDesc &desc) {
  return ge::Tensor(desc);
}

void WriteSymbols(Node *node, const std::vector<uint64_t> &in_syms, const std::vector<uint64_t> &out_syms) {
  if (node == nullptr || node->GetExtendInfo() == nullptr) {
    return;
  }
  auto *extend = node->GetExtendInfo();
  extend->UpdateInputSymbols(in_syms.size());
  extend->UpdateOutputSymbols(out_syms.size());
  for (size_t i = 0; i < in_syms.size(); ++i) {
    extend->SetInputSymbol(i, in_syms[i]);
  }
  for (size_t i = 0; i < out_syms.size(); ++i) {
    extend->SetOutputSymbol(i, out_syms[i]);
  }
}

}  // namespace

Node *Node::InputProducer(int32_t index) const {
  if (builder_ == nullptr) {
    return nullptr;
  }
  const auto in = gnode_.GetInDataNodesAndPortIndexs(index);
  if (in.first == nullptr) {
    return nullptr;
  }
  return builder_->Find(*in.first);
}

FxGraphBuilder::FxGraphBuilder(const std::string &name)
    : es_builder_(std::make_unique<::EsCGraphBuilder>(name.c_str(), static_cast<ge::GraphKind>(1))) {
  if (es_builder_ == nullptr) {
    return;
  }
  graph_ = es_builder_->GetGraph();
  if (graph_ == nullptr) {
    return;
  }
  registered_ = graph_;
  std::lock_guard<std::mutex> lock(RegistryMutex());
  Registry()[registered_] = this;
}

FxGraphBuilder::~FxGraphBuilder() {
  if (registered_ == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(RegistryMutex());
  auto it = Registry().find(registered_);
  if (it != Registry().end() && it->second == this) {
    Registry().erase(it);
  }
}

std::unique_ptr<ge::Graph> FxGraphBuilder::BuildGraphAndReset() {
  auto graph = es_builder_->BuildGraphAndReset();
  graph_ = nullptr;
  return graph;
}

FxGraphBuilder *FxGraphBuilder::Lookup(ge::Graph *graph) {
  if (graph == nullptr) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(RegistryMutex());
  const auto it = Registry().find(graph);
  return it == Registry().end() ? nullptr : it->second;
}

Node *FxGraphBuilder::Find(const ge::GNode &gnode) const {
  ge::AscendString name;
  if (gnode.GetName(name) != ge::GRAPH_SUCCESS || name.GetString() == nullptr) {
    return nullptr;
  }
  const auto it = by_name_.find(name.GetString());
  return it == by_name_.end() ? nullptr : it->second;
}

Node *FxGraphBuilder::Adopt(const ge::GNode &gnode) {
  ge::AscendString name;
  if (gnode.GetName(name) != ge::GRAPH_SUCCESS || name.GetString() == nullptr) {
    return nullptr;
  }
  const std::string key(name.GetString());
  const auto found = by_name_.find(key);
  if (found != by_name_.end()) {
    return found->second;
  }
  ge::AscendString type;
  std::string type_name;
  if (gnode.GetType(type) == ge::GRAPH_SUCCESS && type.GetString() != nullptr) {
    type_name = type.GetString();
  }
  auto owned = std::make_unique<Node>(this, gnode, key, std::move(type_name));
  Node *raw = owned.get();
  owned_.push_back(std::move(owned));
  by_name_.emplace(key, raw);
  return raw;
}

Node *FxGraphBuilder::AddNode(const FxNodeSpecPtr &spec) {
  if (graph_ == nullptr || spec == nullptr) {
    return nullptr;
  }
  ge::es::CompliantNodeBuilder::AnyTypeOperator op(spec->GetName().c_str(), spec->GetType().c_str());
  for (const auto &input : spec->Inputs()) {
    op.InputRegister(input.name.c_str());
  }
  for (const auto &output : spec->Outputs()) {
    op.OutputRegister(output.name.c_str());
  }
  for (const auto &attr : spec->Attrs()) {
    op.AttrRegister(attr.first.c_str(), attr.second);
    op.SetAttr(attr.first.c_str(), attr.second);
  }
  for (const auto &attr : spec->BytesAttrs()) {
    op.AttrRegister(attr.first.c_str(), attr.second);
    op.SetAttr(attr.first.c_str(), attr.second);
  }
  if (const auto *registration = FindFxrtKernelRegistration(spec->GetType()); registration != nullptr) {
    const auto &module_path = GetFxrtKernelModulePath();
    if (module_path.empty() || !FxAttrUtils::SetStr(spec, kCustomKernelNameAttr, registration->kernel_name) ||
        !FxAttrUtils::SetStr(spec, kCustomKernelSoPathAttr, module_path.c_str()) ||
        !FxAttrUtils::SetStr(spec, kCustomRunFuncSymbolAttr, registration->run_func_symbol) ||
        (registration->outputs_creator_symbol != nullptr &&
         !FxAttrUtils::SetStr(spec, kCustomOutputsCreatorSymbolAttr, registration->outputs_creator_symbol))) {
      RT_GLOG(ERROR) << "Failed to attach FXRT kernel registration attributes for " << spec->GetType();
      return nullptr;
    }
    for (const auto &attr : spec->Attrs()) {
      if (attr.first == kCustomKernelNameAttr || attr.first == kCustomKernelSoPathAttr ||
          attr.first == kCustomRunFuncSymbolAttr || attr.first == kCustomOutputsCreatorSymbolAttr) {
        op.AttrRegister(attr.first.c_str(), attr.second);
        op.SetAttr(attr.first.c_str(), attr.second);
      }
    }
  }
  if (spec->GetType() == "FxExecuteOpPrepare") {
    for (const auto *name : {"keep_dims", "transpose_x1", "transpose_x2"}) {
      if (spec->Attrs().find(name) != spec->Attrs().end()) {
        op.RequiredAttrWithTypeRegister(name, "Bool");
      }
    }
  }
  ge::GNode gnode = graph_->AddNodeByOp(op);
  Node *node = Adopt(gnode);
  if (node == nullptr) {
    return nullptr;
  }
  for (size_t i = 0; i < spec->Inputs().size(); ++i) {
    (void)gnode.UpdateInputDesc(static_cast<int32_t>(i), spec->GetInputDesc(i));
  }
  for (size_t i = 0; i < spec->Outputs().size(); ++i) {
    (void)gnode.UpdateOutputDesc(static_cast<int32_t>(i), spec->GetOutputDesc(i));
  }
  return node;
}

Node *FxGraphBuilder::AddFeed(const std::string &name, const char *type, int32_t index,
                              const ge::TensorDesc &tensor_desc) {
  if (type != nullptr && std::string(type) == "Data" && index >= 0) {
    return AddGraphInput(name, type, index, tensor_desc);
  }
  if (graph_ == nullptr) {
    return nullptr;
  }
  ge::es::CompliantNodeBuilder::AnyTypeOperator op(name.c_str(), type == nullptr ? "Data" : type);
  op.OutputRegister("y");
  op.AttrRegister(kFeedIndex, static_cast<int64_t>(index));
  ge::GNode gnode = graph_->AddNodeByOp(op);
  Node *node = Adopt(gnode);
  if (node == nullptr) {
    return nullptr;
  }
  const auto dims = tensor_desc.GetShape().GetDims();
  ge::TensorDesc public_desc(ge::Shape(dims), tensor_desc.GetFormat(), tensor_desc.GetDataType());
  public_desc.SetOriginFormat(tensor_desc.GetOriginFormat());
  public_desc.SetOriginShape(tensor_desc.GetOriginShape());
  (void)gnode.UpdateOutputDesc(0, public_desc);
  if (type != nullptr && std::string(type) == "ConstData") {
    ge::AscendString stage(kStageInit);
    (void)gnode.SetAttr(ge::AscendString(kStageAttr), stage);
  }
  return node;
}

Node *FxGraphBuilder::AddGraphInput(const std::string &name, const char *type, int32_t index,
                                    const ge::TensorDesc &tensor_desc) {
  if (es_builder_ == nullptr) {
    return nullptr;
  }
  const auto dims = tensor_desc.GetShape().GetDims();
  EsCTensorHolder *holder = es_builder_->AddGraphInput(
      index, name.c_str(), type, static_cast<C_DataType>(tensor_desc.GetDataType()),
      static_cast<C_Format>(tensor_desc.GetFormat()), dims.data(), static_cast<int64_t>(dims.size()));
  if (holder == nullptr) {
    return nullptr;
  }
  Node *node = Adopt(holder->GetProducer());
  if (node != nullptr) {
    ge::TensorDesc named(tensor_desc);
    named.SetName(name.c_str());
    model_inputs_.push_back(ToModelIoTensor(named));
    (void)RefreshModelIoAttrs();
  }
  return node;
}

Node *FxGraphBuilder::AddConst(const std::string &name, const void *data, size_t size, bool is_string,
                               FxrtConstType const_type, const char *stage) {
  if (graph_ == nullptr || data == nullptr) {
    return nullptr;
  }
  ge::es::CompliantNodeBuilder::AnyTypeOperator op(name.c_str(), "Const");
  op.OutputRegister("y");
  ge::Operator::OpBytes bytes(reinterpret_cast<const uint8_t *>(data), reinterpret_cast<const uint8_t *>(data) + size);
  op.SetAttr(kConstValue, bytes);
  op.SetAttr("is_string", is_string);
  const int64_t type_value = static_cast<int64_t>(const_type);
  op.AttrRegister(kCustomConstTypeAttr, type_value);
  op.SetAttr(kCustomConstTypeAttr, type_value);
  const ge::AscendString stage_value(stage == nullptr ? kStageInit : stage);
  op.AttrRegister(kStageAttr, stage_value);
  op.SetAttr(kStageAttr, stage_value);
  return Adopt(graph_->AddNodeByOp(op));
}

ge::graphStatus FxGraphBuilder::AddDataEdge(Node *src, int32_t src_port, Node *dst, int32_t dst_port) {
  if (src_port < 0 || dst_port < 0) {
    return AddControlEdge(src, dst);
  }
  if (graph_ == nullptr || src == nullptr || dst == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto status = graph_->AddDataEdge(src->gnode(), src_port, dst->gnode(), dst_port);
  if (status != ge::GRAPH_SUCCESS) {
    RT_GLOG(ERROR) << "ES AddDataEdge failed: " << src->GetName() << "[" << src_port << "] -> " << dst->GetName()
                   << "[" << dst_port << "], source outputs=" << src->OutputCount()
                   << ", destination inputs=" << dst->InputCount();
  }
  return status;
}

ge::graphStatus FxGraphBuilder::AddControlEdge(Node *src, Node *dst) {
  if (graph_ == nullptr || src == nullptr || dst == nullptr) {
    return ge::GRAPH_FAILED;
  }
  return graph_->AddControlEdge(src->gnode(), dst->gnode());
}

std::vector<FxGraphBuilder::DataEdgeInfo> FxGraphBuilder::GetInDataEdges(Node *node) const {
  std::vector<DataEdgeInfo> result;
  if (node == nullptr) {
    return result;
  }
  for (size_t i = 0; i < node->InputCount(); ++i) {
    const auto in = node->gnode().GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
    if (in.first == nullptr) {
      continue;
    }
    auto *src = Find(*in.first);
    if (src != nullptr) {
      result.push_back({src, in.second, node, static_cast<int32_t>(i)});
    }
  }
  return result;
}

std::vector<FxGraphBuilder::DataEdgeInfo> FxGraphBuilder::GetOutDataEdges(Node *node) const {
  std::vector<DataEdgeInfo> result;
  if (node == nullptr) {
    return result;
  }
  for (size_t i = 0; i < node->OutputCount(); ++i) {
    for (const auto &out : node->gnode().GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i))) {
      if (out.first == nullptr) {
        continue;
      }
      auto *dst = Find(*out.first);
      if (dst != nullptr) {
        result.push_back({node, static_cast<int32_t>(i), dst, out.second});
      }
    }
  }
  return result;
}

std::vector<Node *> FxGraphBuilder::GetInControlNodes(Node *node) const {
  std::vector<Node *> result;
  if (node == nullptr) {
    return result;
  }
  for (const auto &peer : node->gnode().GetInControlNodes()) {
    if (peer != nullptr) {
      if (auto *src = Find(*peer); src != nullptr) {
        result.push_back(src);
      }
    }
  }
  return result;
}

std::vector<Node *> FxGraphBuilder::GetOutControlNodes(Node *node) const {
  std::vector<Node *> result;
  if (node == nullptr) {
    return result;
  }
  for (const auto &peer : node->gnode().GetOutControlNodes()) {
    if (peer != nullptr) {
      if (auto *dst = Find(*peer); dst != nullptr) {
        result.push_back(dst);
      }
    }
  }
  return result;
}

std::vector<Node *> FxGraphBuilder::GetAllNodes() const {
  std::vector<Node *> result;
  if (graph_ == nullptr) {
    return result;
  }
  for (const auto &gnode : graph_->GetAllNodes()) {
    if (auto *node = Find(gnode); node != nullptr) {
      result.push_back(node);
    }
  }
  return result;
}

ge::graphStatus FxGraphBuilder::RemoveDataEdge(Node *src, int32_t src_port, Node *dst, int32_t dst_port) {
  if (graph_ == nullptr || src == nullptr || dst == nullptr) {
    return ge::GRAPH_FAILED;
  }
  return graph_->RemoveEdge(src->gnode(), src_port, dst->gnode(), dst_port);
}

ge::graphStatus FxGraphBuilder::RemoveControlEdge(Node *src, Node *dst) {
  return RemoveDataEdge(src, -1, dst, -1);
}

ge::graphStatus FxGraphBuilder::RemoveNode(Node *node) {
  if (graph_ == nullptr || node == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto ret = graph_->RemoveNode(node->gnode());
  if (ret == ge::GRAPH_SUCCESS) {
    by_name_.erase(node->GetName());
  }
  return ret;
}

ge::graphStatus FxGraphBuilder::SetOutputs(const std::vector<std::pair<Node *, int32_t>> &outputs) {
  if (graph_ == nullptr || es_builder_ == nullptr) {
    return ge::GRAPH_FAILED;
  }
  std::vector<std::pair<ge::GNode, int32_t>> public_outputs;
  public_outputs.reserve(outputs.size());
  for (const auto &output : outputs) {
    if (output.first == nullptr) {
      return ge::GRAPH_FAILED;
    }
    public_outputs.emplace_back(output.first->gnode(), output.second);
  }
  const auto ret = graph_->SetOutputs(public_outputs);
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }
  for (size_t i = 0; i < public_outputs.size(); ++i) {
    auto *holder = es_builder_->GetTensorHolderFromNode(public_outputs[i].first, public_outputs[i].second);
    if (holder == nullptr || es_builder_->SetGraphOutput(holder, static_cast<int64_t>(i)) != ge::SUCCESS) {
      return ge::GRAPH_FAILED;
    }
  }
  model_outputs_.clear();
  model_outputs_.reserve(public_outputs.size());
  for (size_t i = 0; i < public_outputs.size(); ++i) {
    ge::TensorDesc desc;
    if (public_outputs[i].first.GetOutputDesc(public_outputs[i].second, desc) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    const std::string output_name = "output_" + std::to_string(i);
    desc.SetName(output_name.c_str());
    model_outputs_.push_back(ToModelIoTensor(desc));
  }
  return RefreshModelIoAttrs();
}

ge::graphStatus FxGraphBuilder::RefreshModelIoAttrs() {
  if (graph_ == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::AttrValue inputs;
  ge::AttrValue outputs;
  ge::AttrValue stream_count;
  ge::AttrValue event_count;
  ge::AttrValue notify_count;
  ge::AttrValue attached_stream_count;
  if (inputs.SetAttrValue(model_inputs_) != ge::GRAPH_SUCCESS ||
      outputs.SetAttrValue(model_outputs_) != ge::GRAPH_SUCCESS ||
      stream_count.SetAttrValue(kModelStreamCountValue) != ge::GRAPH_SUCCESS ||
      event_count.SetAttrValue(static_cast<int64_t>(0)) != ge::GRAPH_SUCCESS ||
      notify_count.SetAttrValue(kModelNotifyCountValue) != ge::GRAPH_SUCCESS ||
      attached_stream_count.SetAttrValue(static_cast<int64_t>(0)) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (graph_->SetAttr(ge::AscendString(kModelInputs), inputs) != ge::GRAPH_SUCCESS ||
      graph_->SetAttr(ge::AscendString(kModelOutputs), outputs) != ge::GRAPH_SUCCESS ||
      graph_->SetAttr(ge::AscendString(kModelStreamCount), stream_count) != ge::GRAPH_SUCCESS ||
      graph_->SetAttr(ge::AscendString(kModelEventCount), event_count) != ge::GRAPH_SUCCESS ||
      graph_->SetAttr(ge::AscendString(kModelNotifyCount), notify_count) != ge::GRAPH_SUCCESS ||
      graph_->SetAttr(ge::AscendString(kModelAttachedStreamCount), attached_stream_count) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FxGraphBuilder::SetNodeIntAttr(Node *node, const char *name, int64_t value) {
  if (node == nullptr || name == nullptr) {
    return ge::GRAPH_FAILED;
  }
  int64_t attr_value = value;
  ge::AscendString key(name);
  return node->gnode().SetAttr(key, attr_value);
}

ge::graphStatus FxGraphBuilder::SetNodeBoolAttr(Node *node, const char *name, bool value) {
  if (node == nullptr || name == nullptr) {
    return ge::GRAPH_FAILED;
  }
  return node->gnode().SetAttr(ge::AscendString(name), value);
}

ge::graphStatus FxGraphBuilder::SetNodeStringAttr(Node *node, const char *name, const char *value) {
  if (node == nullptr || name == nullptr || value == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ge::AscendString key(name);
  ge::AscendString attr_value(value);
  return node->gnode().SetAttr(key, attr_value);
}

ge::graphStatus FxGraphBuilder::SetNodeBytesAttr(Node *node, const char *name, const uint8_t *data, size_t size) {
  if (node == nullptr || name == nullptr || (data == nullptr && size != 0U)) {
    return ge::GRAPH_FAILED;
  }
  ge::OpBytes attr_value;
  if (size > 0U) {
    attr_value.assign(data, data + size);
  }
  ge::AscendString key(name);
  return node->gnode().SetAttr(key, attr_value);
}

ge::graphStatus FxGraphBuilder::RetainModelBufferData(std::shared_ptr<ge::ModelBufferData> model_buffer) {
  if (model_buffer == nullptr) {
    return ge::GRAPH_PARAM_INVALID;
  }
  model_buffer_constants_.push_back(std::move(model_buffer));
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus FxGraphBuilder::AddInputNode(Node *) { return ge::GRAPH_SUCCESS; }

ge::graphStatus FxGraphBuilder::DumpReadable(const std::string &suffix) const {
  if (graph_ == nullptr) {
    return ge::GRAPH_FAILED;
  }
  return graph_->DumpToFile(ge::Graph::DumpFormat::kReadable, ge::AscendString(suffix.c_str()));
}

Node *AddNode(ge::Graph *graph, const FxNodeSpecPtr &spec) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? nullptr : adapter->AddNode(spec);
}

ge::graphStatus AddDataEdge(ge::Graph *graph, Node *src, int32_t src_port, Node *dst, int32_t dst_port) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? ge::GRAPH_FAILED : adapter->AddDataEdge(src, src_port, dst, dst_port);
}

ge::graphStatus AddControlEdge(ge::Graph *graph, Node *src, Node *dst) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? ge::GRAPH_FAILED : adapter->AddControlEdge(src, dst);
}

ge::graphStatus AddInputNode(ge::Graph *graph, Node *node) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? ge::GRAPH_FAILED : adapter->AddInputNode(node);
}

std::vector<FxGraphBuilder::DataEdgeInfo> GetInDataEdges(ge::Graph *graph, Node *node) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? std::vector<FxGraphBuilder::DataEdgeInfo>{} : adapter->GetInDataEdges(node);
}

std::vector<FxGraphBuilder::DataEdgeInfo> GetOutDataEdges(ge::Graph *graph, Node *node) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? std::vector<FxGraphBuilder::DataEdgeInfo>{} : adapter->GetOutDataEdges(node);
}

std::vector<Node *> GetInControlNodes(ge::Graph *graph, Node *node) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? std::vector<Node *>{} : adapter->GetInControlNodes(node);
}

std::vector<Node *> GetOutControlNodes(ge::Graph *graph, Node *node) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? std::vector<Node *>{} : adapter->GetOutControlNodes(node);
}

std::vector<Node *> GetAllNodes(ge::Graph *graph) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? std::vector<Node *>{} : adapter->GetAllNodes();
}

ge::graphStatus RemoveDataEdge(ge::Graph *graph, Node *src, int32_t src_port, Node *dst, int32_t dst_port) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? ge::GRAPH_FAILED : adapter->RemoveDataEdge(src, src_port, dst, dst_port);
}

ge::graphStatus RemoveControlEdge(ge::Graph *graph, Node *src, Node *dst) {
  return RemoveDataEdge(graph, src, -1, dst, -1);
}

ge::graphStatus RemoveNode(ge::Graph *graph, Node *node) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? ge::GRAPH_FAILED : adapter->RemoveNode(node);
}

ge::graphStatus SetOutputs(ge::Graph *graph, const std::vector<std::pair<Node *, int32_t>> &outputs) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? ge::GRAPH_FAILED : adapter->SetOutputs(outputs);
}

ge::graphStatus SetNodeIntAttr(ge::Graph *graph, Node *node, const char *name, int64_t value) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? ge::GRAPH_FAILED : adapter->SetNodeIntAttr(node, name, value);
}

ge::graphStatus SetNodeBoolAttr(ge::Graph *graph, Node *node, const char *name, bool value) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? ge::GRAPH_FAILED : adapter->SetNodeBoolAttr(node, name, value);
}
ge::graphStatus SetNodeStringAttr(ge::Graph *graph, Node *node, const char *name, const char *value) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? ge::GRAPH_FAILED : adapter->SetNodeStringAttr(node, name, value);
}

ge::graphStatus SetNodeBytesAttr(ge::Graph *graph, Node *node, const char *name, const uint8_t *data, size_t size) {
  auto *adapter = FxGraphBuilder::Lookup(graph);
  return adapter == nullptr ? ge::GRAPH_FAILED : adapter->SetNodeBytesAttr(node, name, data, size);
}

void WireSymbols(Node *node, const std::vector<uint64_t> &in_syms, const std::vector<uint64_t> &out_syms) {
  WriteSymbols(node, in_syms, out_syms);
}

uint64_t GetInputSymbol(const Node *node, size_t index) {
  return node == nullptr || node->GetExtendInfo() == nullptr ? kInvalidSymbol
                                                             : node->GetExtendInfo()->GetInputSymbol(index);
}

uint64_t GetOutputSymbol(const Node *node, size_t index) {
  return node == nullptr || node->GetExtendInfo() == nullptr ? kInvalidSymbol
                                                             : node->GetExtendInfo()->GetOutputSymbol(index);
}

}  // namespace fx_converter
