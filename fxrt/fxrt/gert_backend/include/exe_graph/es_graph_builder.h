#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "exe_graph/fx_node.h"
#include "exe_graph/fx_node_spec.h"
#include "exe_graph/const_value_serialization.h"
#include "ge/es_c_graph_builder.h"
#include "graph/graph.h"

namespace ge {
struct ModelBufferData;
}

namespace fx_converter {

// Non-owning view of the public graph. get() is the pointer lowerings already pass around.
struct EsGraphRef {
  ge::Graph* graph = nullptr;

  ge::Graph* get() const {
    return graph;
  }
  ge::Graph* operator->() const {
    return graph;
  }
  explicit operator bool() const {
    return graph != nullptr;
  }
  EsGraphRef& operator=(ge::Graph* pointer) {
    graph = pointer;
    return *this;
  }
};

inline bool operator==(const EsGraphRef& lhs, const EsGraphRef& rhs) {
  return lhs.graph == rhs.graph;
}
inline bool operator!=(const EsGraphRef& lhs, const EsGraphRef& rhs) {
  return lhs.graph != rhs.graph;
}
inline bool operator==(const EsGraphRef& lhs, std::nullptr_t) {
  return lhs.graph == nullptr;
}
inline bool operator!=(const EsGraphRef& lhs, std::nullptr_t) {
  return lhs.graph != nullptr;
}
inline bool operator==(std::nullptr_t, const EsGraphRef& rhs) {
  return rhs.graph == nullptr;
}
inline bool operator!=(std::nullptr_t, const EsGraphRef& rhs) {
  return rhs.graph != nullptr;
}

// Nodes and edges are created through the public ES Graph and GNode APIs.
class FxGraphBuilder {
 public:
  struct DataEdgeInfo {
    Node* src = nullptr;
    int32_t src_port = -1;
    Node* dst = nullptr;
    int32_t dst_port = -1;
  };

  explicit FxGraphBuilder(const std::string& name);
  ~FxGraphBuilder();

  FxGraphBuilder(const FxGraphBuilder&) = delete;
  FxGraphBuilder& operator=(const FxGraphBuilder&) = delete;

  ge::Graph* GetGraph() const {
    return graph_;
  }
  std::unique_ptr<ge::Graph> BuildGraphAndReset();

  Node* AddNode(const FxNodeSpecPtr& spec);
  Node* AddFeed(const std::string& name, const char* type, int32_t index, const ge::TensorDesc& tensor_desc);
  Node* AddGraphInput(const std::string& name, const char* type, int32_t index, const ge::TensorDesc& tensor_desc);
  Node* AddConst(
      const std::string& name,
      const void* data,
      size_t size,
      bool is_string,
      FxrtConstType const_type = FxrtConstType::kBytePtr,
      const char* stage = nullptr);

  ge::graphStatus AddDataEdge(Node* src, int32_t src_port, Node* dst, int32_t dst_port);
  ge::graphStatus AddControlEdge(Node* src, Node* dst);
  std::vector<DataEdgeInfo> GetInDataEdges(Node* node) const;
  std::vector<DataEdgeInfo> GetOutDataEdges(Node* node) const;
  std::vector<Node*> GetInControlNodes(Node* node) const;
  std::vector<Node*> GetOutControlNodes(Node* node) const;
  std::vector<Node*> GetAllNodes() const;
  ge::graphStatus RemoveDataEdge(Node* src, int32_t src_port, Node* dst, int32_t dst_port);
  ge::graphStatus RemoveControlEdge(Node* src, Node* dst);
  ge::graphStatus RemoveNode(Node* node);

  ge::graphStatus SetOutputs(const std::vector<std::pair<Node*, int32_t>>& outputs);
  ge::graphStatus SetNodeIntAttr(Node* node, const char* name, int64_t value);
  ge::graphStatus SetNodeBoolAttr(Node* node, const char* name, bool value);
  ge::graphStatus SetNodeStringAttr(Node* node, const char* name, const char* value);
  ge::graphStatus SetNodeBytesAttr(Node* node, const char* name, const uint8_t* data, size_t size);
  ge::graphStatus RetainModelBufferData(std::shared_ptr<ge::ModelBufferData> model_buffer);

  ge::graphStatus AddInputNode(Node* node);
  ge::graphStatus DumpReadable(const std::string& suffix) const;

  static FxGraphBuilder* Lookup(ge::Graph* graph);
  Node* Find(const ge::GNode& gnode) const;

 private:
  Node* Adopt(const ge::GNode& gnode);
  ge::graphStatus RefreshModelIoAttrs();

  std::unique_ptr<::EsCGraphBuilder> es_builder_;
  ge::Graph* graph_ = nullptr;
  ge::Graph* registered_ = nullptr;
  std::list<std::unique_ptr<Node>> owned_;
  std::unordered_map<std::string, Node*> by_name_;
  std::vector<ge::Tensor> model_inputs_;
  std::vector<ge::Tensor> model_outputs_;
  std::vector<std::shared_ptr<ge::ModelBufferData>> model_buffer_constants_;
};

Node* AddNode(ge::Graph* graph, const FxNodeSpecPtr& spec);
ge::graphStatus AddDataEdge(ge::Graph* graph, Node* src, int32_t src_port, Node* dst, int32_t dst_port);
ge::graphStatus AddControlEdge(ge::Graph* graph, Node* src, Node* dst);
ge::graphStatus AddInputNode(ge::Graph* graph, Node* node);
std::vector<FxGraphBuilder::DataEdgeInfo> GetInDataEdges(ge::Graph* graph, Node* node);
std::vector<FxGraphBuilder::DataEdgeInfo> GetOutDataEdges(ge::Graph* graph, Node* node);
std::vector<Node*> GetInControlNodes(ge::Graph* graph, Node* node);
std::vector<Node*> GetOutControlNodes(ge::Graph* graph, Node* node);
std::vector<Node*> GetAllNodes(ge::Graph* graph);
ge::graphStatus RemoveDataEdge(ge::Graph* graph, Node* src, int32_t src_port, Node* dst, int32_t dst_port);
ge::graphStatus RemoveControlEdge(ge::Graph* graph, Node* src, Node* dst);
ge::graphStatus RemoveNode(ge::Graph* graph, Node* node);
ge::graphStatus SetOutputs(ge::Graph* graph, const std::vector<std::pair<Node*, int32_t>>& outputs);
ge::graphStatus SetNodeIntAttr(ge::Graph* graph, Node* node, const char* name, int64_t value);
ge::graphStatus SetNodeBoolAttr(ge::Graph* graph, Node* node, const char* name, bool value);
ge::graphStatus SetNodeStringAttr(ge::Graph* graph, Node* node, const char* name, const char* value);
ge::graphStatus SetNodeBytesAttr(ge::Graph* graph, Node* node, const char* name, const uint8_t* data, size_t size);

void WireSymbols(Node* node, const std::vector<uint64_t>& in_syms, const std::vector<uint64_t>& out_syms);
uint64_t GetInputSymbol(const Node* node, size_t index);
uint64_t GetOutputSymbol(const Node* node, size_t index);

} // namespace fx_converter
