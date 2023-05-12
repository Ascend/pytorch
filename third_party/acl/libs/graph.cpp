#include "graph/graph.h"
#include "graph/ascend_string.h"
#include "graph/tensor.h"

namespace ge {
Graph::Graph(const char* name) {}

Graph graph("test");

Graph& Graph::SetInputs(const std::vector<Operator>& inputs) {
  return graph;
}

Graph& Graph::SetOutputs(const std::vector<Operator>& outputs) {
  return graph;
}

Graph& Graph::SetOutputs(
    const std::vector<std::pair<Operator, std::vector<size_t>>>&
        output_indexs) {
  return graph;
}
AscendString::AscendString(const char* name) {}
bool AscendString::operator<(const AscendString& d) const {
  return true;
}
const char* AscendString::GetString() const {
  return 0;
}
bool AscendString::operator==(const AscendString& d) const {
  return true;
}

TensorDesc::TensorDesc(const TensorDesc& desc) {}
TensorDesc::TensorDesc(TensorDesc&& desc) {}
TensorDesc::TensorDesc(Shape shape, Format format, DataType dt) {}
void TensorDesc::SetConstData(
    const std::shared_ptr<void> const_data_buffer,
    const size_t& const_data_len) {}
Shape::Shape(const std::vector<int64_t>& dims) {}

std::vector<GNode> Graph::GetDirectNode() const {
  return {};
}

GNode::GNode() {}
graphStatus GNode::GetType(AscendString& type) const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, int64_t& attr_value)
    const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, int32_t& attr_value)
    const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, float& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, AscendString& attr_value)
    const {
  return 0;
}
graphStatus GNode::SetAttr(const AscendString& name, bool& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(
    const AscendString& name,
    std::vector<int64_t>& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(
    const AscendString& name,
    std::vector<float>& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(
    const AscendString& name,
    std::vector<bool>& attr_value) const {
  return 0;
}
graphStatus GNode::SetAttr(
    const AscendString& name,
    std::vector<std::vector<int64_t>>& attr_value) const {
  return 0;
}
} // namespace ge