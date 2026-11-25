// Link-only stubs. Never use this library at runtime.
#include "../cann/include/graph/operator.h"
#include "../cann/include/graph/tensor.h"
#include "../cann/include/graph/types.h"
#include "../cann/include/graph/utils/type_utils.h"
#include "../ge/include/inc/graph_metadef/graph/buffer.h"
#include "../ge/include/inc/graph_metadef/graph/compute_graph.h"
#include "../ge/include/inc/graph_metadef/graph/fast_graph/execute_graph.h"
#include "../ge/include/inc/graph_metadef/graph/fast_graph/fast_node.h"
#include "../ge/include/inc/graph_metadef/graph/ge_tensor.h"
#include "../ge/include/inc/graph_metadef/graph/node.h"
#include "../ge/include/inc/graph_metadef/graph/op_desc.h"
#include "../ge/include/inc/graph_metadef/graph/utils/execute_graph_adapter.h"
#include "../ge/include/inc/graph_metadef/graph/utils/execute_graph_utils.h"
#include "../ge/include/inc/graph_metadef/graph/utils/graph_utils.h"

extern "C" void fxrt_cann_stub_marker() {
  __builtin_trap();
}

namespace ge {
auto GetSizeInBytes(int64_t, DataType) -> int64_t {
  __builtin_trap();
}
Shape::Shape() {
  __builtin_trap();
}
Shape::Shape(const std::vector<int64_t> &) {
  __builtin_trap();
}
auto Shape::GetDims() const -> std::vector<int64_t> {
  __builtin_trap();
}
Operator::Operator(const char_t *) {
  __builtin_trap();
}
Operator::Operator(const char_t *, const char_t *) {
  __builtin_trap();
}
}  // namespace ge
namespace ge {
TensorDesc::TensorDesc() {
  __builtin_trap();
}
TensorDesc::TensorDesc(Shape, Format, DataType) {
  __builtin_trap();
}
TensorDesc::TensorDesc(const TensorDesc &) {
  __builtin_trap();
}
TensorDesc::TensorDesc(TensorDesc &&) {
  __builtin_trap();
}
auto TensorDesc::operator=(TensorDesc &&) -> TensorDesc & {
  __builtin_trap();
}
auto TensorDesc::GetShape() const -> Shape {
  __builtin_trap();
}
auto TensorDesc::GetFormat() const -> Format {
  __builtin_trap();
}
auto TensorDesc::GetOriginShape() const -> Shape {
  __builtin_trap();
}
auto TensorDesc::SetOriginShape(const Shape &) -> void {
  __builtin_trap();
}
auto TensorDesc::GetOriginFormat() const -> Format {
  __builtin_trap();
}
auto TensorDesc::SetOriginFormat(const Format) -> void {
  __builtin_trap();
}
auto TensorDesc::SetName(const char_t *) -> void {
  __builtin_trap();
}
auto TensorDesc::GetDataType() const -> DataType {
  __builtin_trap();
}
Tensor::Tensor(const TensorDesc &) {
  __builtin_trap();
}
auto Tensor::SetData(const std::vector<uint8_t> &) -> graphStatus {
  __builtin_trap();
}
auto Operator::SetAttr(const char_t *, int64_t) -> Operator & {
  __builtin_trap();
}
auto Operator::SetAttr(const char_t *, const AscendString &) -> Operator & {
  __builtin_trap();
}
auto Operator::GetAttr(const char_t *, int64_t &) const -> graphStatus {
  __builtin_trap();
}
auto Operator::GetAttr(const char_t *, float32_t &) const -> graphStatus {
  __builtin_trap();
}
auto Operator::SetAttr(const char_t *, const AttrValue &) -> Operator & {
  __builtin_trap();
}
auto Operator::SetAttr(const char_t *, bool) -> Operator & {
  __builtin_trap();
}
auto Operator::SetAttr(const char_t *, const OpBytes &) -> Operator & {
  __builtin_trap();
}
auto Operator::AttrRegister(const char_t *, const AttrValue &) -> void {
  __builtin_trap();
}
auto Operator::AttrRegister(const char_t *, int64_t) -> void {
  __builtin_trap();
}
auto Operator::AttrRegister(const char_t *, const AscendString &) -> void {
  __builtin_trap();
}
auto Operator::AttrRegister(const char_t *, const OpBytes &) -> void {
  __builtin_trap();
}
auto Operator::AttrRegister(const char_t *, bool) -> void {
  __builtin_trap();
}
auto Operator::InputRegister(const char_t *) -> void {
  __builtin_trap();
}
auto Operator::OutputRegister(const char_t *) -> void {
  __builtin_trap();
}
auto Operator::RequiredAttrWithTypeRegister(const char_t *, const char_t *) -> void {
  __builtin_trap();
}
auto TypeUtils::SerialStringToDataType(const std::string &) -> DataType {
  __builtin_trap();
}
Buffer::Buffer() {
  __builtin_trap();
}
Buffer::~Buffer() {
  __builtin_trap();
}
auto Buffer::CopyFrom(const std::uint8_t *const, const std::size_t) -> Buffer {
  __builtin_trap();
}
auto Buffer::GetData() -> std::uint8_t * {
  __builtin_trap();
}
auto Buffer::data() -> std::uint8_t * {
  __builtin_trap();
}
auto Buffer::GetSize() const -> std::size_t {
  __builtin_trap();
}
auto Buffer::size() const -> std::size_t {
  __builtin_trap();
}
GeShape::GeShape() {
  __builtin_trap();
}
GeShape::~GeShape() {
  __builtin_trap();
}
GeShape::GeShape(std::vector<int64_t>) {
  __builtin_trap();
}
auto GeShape::GetDims() const -> std::vector<int64_t> {
  __builtin_trap();
}
GeTensorDesc::GeTensorDesc() {
  __builtin_trap();
}
GeTensorDesc::GeTensorDesc(const GeShape &, const Format, const DataType) {
  __builtin_trap();
}
GeTensorDesc::GeTensorDesc(const GeTensorDesc &) {
  __builtin_trap();
}
GeTensorDesc::GeTensorDesc(GeTensorDesc &&) {
  __builtin_trap();
}
GeTensorDesc::~GeTensorDesc() noexcept {
  __builtin_trap();
}
auto GeTensorDesc::GetShape() const -> const GeShape & {
  __builtin_trap();
}
auto GeTensorDesc::SetOriginShape(const GeShape &) -> void {
  __builtin_trap();
}
auto GeTensorDesc::GetFormat() const -> Format {
  __builtin_trap();
}
auto GeTensorDesc::GetOriginFormat() const -> Format {
  __builtin_trap();
}
auto GeTensorDesc::SetOriginFormat(const Format) -> void {
  __builtin_trap();
}
auto GeTensorDesc::GetDataType() const -> DataType {
  __builtin_trap();
}
auto GeTensorDesc::operator=(GeTensorDesc &&) -> GeTensorDesc & {
  __builtin_trap();
}
GeTensor::GeTensor(const GeTensorDesc &, const std::vector<uint8_t> &) {
  __builtin_trap();
}
GeTensor::~GeTensor() {
  __builtin_trap();
}
OpDesc::OpDesc(const std::string &, const std::string &) {
  __builtin_trap();
}
auto OpDesc::AddInputDesc(const GeTensorDesc &) -> graphStatus {
  __builtin_trap();
}
auto OpDesc::AddInputDesc(const std::string &, const GeTensorDesc &) -> graphStatus {
  __builtin_trap();
}
auto OpDesc::GetInputsSize() const -> size_t {
  __builtin_trap();
}
auto OpDesc::AddOutputDesc(const GeTensorDesc &) -> graphStatus {
  __builtin_trap();
}
auto OpDesc::AddOutputDesc(const std::string &, const GeTensorDesc &) -> graphStatus {
  __builtin_trap();
}
auto OpDesc::GetOutputDesc(const uint32_t) const -> const GeTensorDesc & {
  __builtin_trap();
}
auto OpDesc::GetOutputsSize() const -> size_t {
  __builtin_trap();
}
auto OpDesc::AppendIrAttrName(const std::string &) -> void {
  __builtin_trap();
}
auto OpDesc::AppendIrInput(std::string, IrInputType) -> void {
  __builtin_trap();
}
auto OpDesc::AppendIrOutput(std::string, IrOutputType) -> void {
  __builtin_trap();
}
auto OpDesc::SetOpInferDepends(const std::vector<std::string> &) -> void {
  __builtin_trap();
}
auto OpDesc::GetInputIndexByName(const std::string &) const -> int32_t {
  __builtin_trap();
}
auto OpDesc::AddSubgraphName(const std::string &) -> graphStatus {
  __builtin_trap();
}
auto Node::GetInDataAnchor(const int32_t) const -> InDataAnchorPtr {
  __builtin_trap();
}
auto Node::GetOutDataAnchor(const int32_t) const -> OutDataAnchorPtr {
  __builtin_trap();
}
auto ExtendInfo::UpdateInputSymbols(size_t) -> void {
  __builtin_trap();
}
auto ExtendInfo::UpdateOutputSymbols(size_t) -> void {
  __builtin_trap();
}
auto ExtendInfo::SetInputSymbol(size_t, uint64_t) -> graphStatus {
  __builtin_trap();
}
auto ExtendInfo::SetOutputSymbol(size_t, uint64_t) -> graphStatus {
  __builtin_trap();
}
auto ExtendInfo::GetInputSymbol(size_t) -> uint64_t {
  __builtin_trap();
}
auto ExtendInfo::GetOutputSymbol(size_t) -> uint64_t {
  __builtin_trap();
}
auto FastNode::GetOpDescBarePtr() const -> OpDesc * {
  __builtin_trap();
}
auto FastNode::GetOpDescPtr() const -> OpDescPtr {
  __builtin_trap();
}
auto FastNode::GetType() const -> std::string {
  __builtin_trap();
}
auto FastNode::GetTypePtr() const -> const char * {
  __builtin_trap();
}
auto FastNode::GetName() const -> std::string {
  __builtin_trap();
}
auto FastNode::GetNamePtr() const -> const char * {
  __builtin_trap();
}
auto FastNode::GetDataInNum() const -> size_t {
  __builtin_trap();
}
auto FastNode::GetDataOutNum() const -> size_t {
  __builtin_trap();
}
auto FastNode::GetAllInDataEdgesRef() const -> const std::vector<Edge<FastNode> *> & {
  __builtin_trap();
}
auto FastNode::GetInDataEdgeByIndex(int32_t) const -> Edge<FastNode> * {
  __builtin_trap();
}
auto FastNode::GetExtendInfo() const -> ExtendInfo * {
  __builtin_trap();
}
auto FastNode::GetInDataNodes() const -> std::vector<FastNode *> {
  __builtin_trap();
}
auto FastNode::GetInControlNodes() const -> std::vector<FastNode *> {
  __builtin_trap();
}
ExecuteGraph::ExecuteGraph(const std::string &) {
  __builtin_trap();
}
auto ExecuteGraph::MutableAttrMap() -> ProtoAttrMap & {
  __builtin_trap();
}
auto ExecuteGraph::GetAttrMap() const -> ConstProtoAttrMap & {
  __builtin_trap();
}
auto ExecuteGraph::AddNode(const OpDescPtr &) -> FastNode * {
  __builtin_trap();
}
auto ExecuteGraph::AddInputNode(FastNode *const) -> FastNode * {
  __builtin_trap();
}
auto ExecuteGraph::AddEdge(FastNode *const, int32_t, FastNode *const, int32_t) -> FastEdge * {
  __builtin_trap();
}
auto ExecuteGraph::GetDirectNode() const -> std::vector<FastNode *> {
  __builtin_trap();
}
auto ExecuteGraph::GetName() const -> std::string {
  __builtin_trap();
}
auto ExecuteGraph::TopologicalSorting() -> graphStatus {
  __builtin_trap();
}
auto ExecuteGraph::GetAllNodes() const -> std::vector<FastNode *> {
  __builtin_trap();
}
ComputeGraph::ComputeGraph(const std::string &) {
  __builtin_trap();
}
ComputeGraph::ComputeGraph(const char_t *) {
  __builtin_trap();
}
auto ComputeGraph::AddNode(const OpDescPtr) -> NodePtr {
  __builtin_trap();
}
auto ComputeGraph::GetAllSubgraphs() const -> std::vector<std::shared_ptr<ComputeGraph>> {
  __builtin_trap();
}
auto ExecuteGraphAdapter::ConvertExecuteGraphToComputeGraph(ExecuteGraph *) -> ComputeGraphPtr {
  __builtin_trap();
}
auto ExecuteGraphUtils::FindFirstNodeMatchType(ExecuteGraph *, const char_t *const) -> FastNode * {
  __builtin_trap();
}
auto GraphUtils::AddEdge(const OutDataAnchorPtr &, const InDataAnchorPtr &) -> graphStatus {
  __builtin_trap();
}
auto OpDesc::GetName() const -> std::string { __builtin_trap(); }
auto OpDesc::GetType() const -> std::string { __builtin_trap(); }
auto OpDesc::GetAllInputsSize() const -> size_t { __builtin_trap(); }
auto OpDesc::GetInputDesc(const uint32_t) const -> const GeTensorDesc & { __builtin_trap(); }
auto OpDesc::GetInputNameByIndex(const uint32_t) const -> std::string { __builtin_trap(); }
auto OpDesc::GetOutputNameByIndex(const uint32_t) const -> std::string { __builtin_trap(); }
auto OpDesc::GetIrAttrNames() const -> const std::vector<std::string> & { __builtin_trap(); }
auto OpDesc::GetIrInputs() const -> const std::vector<std::pair<std::string, IrInputType>> & {
  __builtin_trap();
}
auto OpDesc::GetIrOutputs() const -> const std::vector<std::pair<std::string, IrOutputType>> & {
  __builtin_trap();
}
auto OpDesc::GetOpInferDepends() const -> std::vector<std::string> { __builtin_trap(); }
auto AnyValue::GetValueType() const noexcept -> ValueType { __builtin_trap(); }
auto AnyValue::GetAddr() const -> const void * { __builtin_trap(); }
template <> TypeId GetTypeId<bool>() { __builtin_trap(); }
template <> TypeId GetTypeId<int64_t>() { __builtin_trap(); }
template <> TypeId GetTypeId<float>() { __builtin_trap(); }
template <> TypeId GetTypeId<std::string>() { __builtin_trap(); }
template <> TypeId GetTypeId<std::vector<int64_t>>() { __builtin_trap(); }
template <> TypeId GetTypeId<std::vector<float>>() { __builtin_trap(); }
template <> TypeId GetTypeId<std::vector<bool>>() { __builtin_trap(); }
template <> TypeId GetTypeId<std::vector<std::string>>() { __builtin_trap(); }
}  // namespace ge
