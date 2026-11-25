// Link-only stubs. Never use this library at runtime.
#include "../cann/include/graph/ascend_string.h"
#include "../cann/include/graph/attr_value.h"
#include "../cann/include/graph/gnode.h"
#include "../cann/include/graph/graph.h"
#include "../cann/include/graph/operator_factory.h"
#include "../ge/include/inc/graph_metadef/graph/utils/graph_utils_ex.h"

extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}

namespace ge {
AscendString::AscendString(const char_t *const) {
  __builtin_trap();
}
auto AscendString::GetString() const -> const char_t * {
  __builtin_trap();
}
auto AscendString::operator<(const AscendString &) const -> bool {
  __builtin_trap();
}
GNode::GNode() {
  __builtin_trap();
}
auto GNode::GetType(AscendString &) const -> graphStatus {
  __builtin_trap();
}
auto GNode::GetName(AscendString &) const -> graphStatus {
  __builtin_trap();
}
auto GNode::GetInDataNodesAndPortIndexs(const int32_t) const -> std::pair<GNodePtr, int32_t> {
  __builtin_trap();
}
auto GNode::GetInControlNodes() const -> std::vector<GNodePtr> {
  __builtin_trap();
}
auto GNode::GetOutDataNodesAndPortIndexs(const int32_t) const -> std::vector<std::pair<GNodePtr, int32_t>> {
  __builtin_trap();
}
auto GNode::GetOutControlNodes() const -> std::vector<GNodePtr> {
  __builtin_trap();
}
auto GNode::GetInputsSize() const -> size_t {
  __builtin_trap();
}
auto GNode::GetOutputsSize() const -> size_t {
  __builtin_trap();
}
auto GNode::UpdateInputDesc(const int32_t, const TensorDesc &) -> graphStatus {
  __builtin_trap();
}
auto GNode::GetOutputDesc(const int32_t, TensorDesc &) const -> graphStatus {
  __builtin_trap();
}
auto GNode::UpdateOutputDesc(const int32_t, const TensorDesc &) -> graphStatus {
  __builtin_trap();
}
auto GNode::GetAttr(const AscendString &, int64_t &) const -> graphStatus {
  __builtin_trap();
}
auto GNode::GetAttr(const AscendString &, OpBytes &) const -> graphStatus {
  __builtin_trap();
}
auto GNode::GetAttr(const AscendString &, AscendString &) const -> graphStatus {
  __builtin_trap();
}
auto GNode::SetAttr(const AscendString &, int64_t &) const -> graphStatus {
  __builtin_trap();
}
auto GNode::SetAttr(const AscendString &, AscendString &) const -> graphStatus {
  __builtin_trap();
}
auto GNode::SetAttr(const AscendString &, OpBytes &) const -> graphStatus {
  __builtin_trap();
}
auto GNode::SetAttr(const AscendString &, bool &) const -> graphStatus {
  __builtin_trap();
}
auto GNode::GetAttr(const AscendString &, bool &) const -> graphStatus {
  __builtin_trap();
}
auto Graph::GetName() const -> const std::string & {
  __builtin_trap();
}
auto Graph::SetOutputs(const std::vector<std::pair<GNode, int32_t>> &) -> graphStatus {
  __builtin_trap();
}
auto Graph::GetAllNodes() const -> std::vector<GNode> {
  __builtin_trap();
}
auto Graph::RemoveNode(GNode &) -> graphStatus {
  __builtin_trap();
}
auto Graph::RemoveEdge(GNode &, const int32_t, GNode &, const int32_t) -> graphStatus {
  __builtin_trap();
}
auto Graph::AddNodeByOp(const Operator &) -> GNode {
  __builtin_trap();
}
auto Graph::AddDataEdge(GNode &, const int32_t, GNode &, const int32_t) -> graphStatus {
  __builtin_trap();
}
auto Graph::AddControlEdge(GNode &, GNode &) -> graphStatus {
  __builtin_trap();
}
auto Graph::SetAttr(const AscendString &, const AttrValue &) -> graphStatus {
  __builtin_trap();
}
auto Graph::DumpToFile(DumpFormat, const AscendString &) const -> graphStatus {
  __builtin_trap();
}
AttrValue::AttrValue() {
  __builtin_trap();
}
auto AttrValue::SetAttrValue(const Tensor &) const -> graphStatus {
  __builtin_trap();
}
auto AttrValue::SetAttrValue(const int64_t &) const -> graphStatus {
  __builtin_trap();
}
auto AttrValue::SetAttrValue(const float32_t &) const -> graphStatus {
  __builtin_trap();
}
auto AttrValue::SetAttrValue(const AscendString &) const -> graphStatus {
  __builtin_trap();
}
auto AttrValue::SetAttrValue(const bool &) const -> graphStatus {
  __builtin_trap();
}
auto AttrValue::SetAttrValue(const std::vector<int64_t> &) const -> graphStatus {
  __builtin_trap();
}
auto AttrValue::SetAttrValue(const std::vector<AscendString> &) const -> graphStatus {
  __builtin_trap();
}
auto AttrValue::SetAttrValue(const std::vector<Tensor> &) const -> graphStatus {
  __builtin_trap();
}
auto OperatorFactory::IsExistOp(const char_t *const) -> bool {
  __builtin_trap();
}
OperatorCreatorRegister::OperatorCreatorRegister(const char_t *const, const OpCreatorV2 &) {
  __builtin_trap();
}
auto GraphUtilsEx::Graph2ExecuteGraph(const Graph &) -> std::shared_ptr<ExecuteGraph> {
  __builtin_trap();
}
}  // namespace ge
