#ifndef INC_EXTERNAL_GRAPH_NODE_H_
#define INC_EXTERNAL_GRAPH_NODE_H_

#include <vector>
#include <cstdint>

#include "./ge_error_codes.h"
#include "./types.h"
#include "./tensor.h"
#include "./ascend_string.h"

namespace ge {
class AttrValue;
class GNode;
class OpDesc;
class Graph;
class ComputeGraph;
using GNodePtr = std::shared_ptr<GNode>;
using GraphPtr = std::shared_ptr<Graph>;
using OpBytes = std::vector<uint8_t>;
using OpDescPtr = std::shared_ptr<OpDesc>;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;

class NodeImpl;
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GNode {
 public:
  GNode();

  ~GNode() = default;

  graphStatus GetType(AscendString &type) const;

  graphStatus GetName(AscendString &name) const;

  std::pair<GNodePtr, int32_t> GetInDataNodesAndPortIndexs(const int32_t index) const;

  std::vector<GNodePtr> GetInControlNodes() const;

  std::vector<std::pair<GNodePtr, int32_t>> GetOutDataNodesAndPortIndexs(const int32_t index) const;

  std::vector<GNodePtr> GetOutControlNodes() const;

  graphStatus GetInputConstData(const int32_t index, Tensor &data) const;

  graphStatus GetInputIndexByName(const AscendString &name, int32_t &index);

  graphStatus GetOutputIndexByName(const AscendString &name, int32_t &index);

  size_t GetInputsSize() const;

  size_t GetOutputsSize() const;

  graphStatus GetInputDesc(const int32_t index, TensorDesc &tensor_desc) const;

  graphStatus UpdateInputDesc(const int32_t index, const TensorDesc &tensor_desc);

  graphStatus GetOutputDesc(const int32_t index, TensorDesc &tensor_desc) const;

  graphStatus UpdateOutputDesc(const int32_t index, const TensorDesc &tensor_desc);

  graphStatus GetAttr(const AscendString &name, int64_t &attr_value) const;
  graphStatus GetAttr(const AscendString &name, int32_t &attr_value) const;
  graphStatus GetAttr(const AscendString &name, uint32_t &attr_value) const;
  graphStatus GetAttr(const AscendString &name, float &attr_value) const;
  graphStatus GetAttr(const AscendString &name, AscendString &attr_value) const;
  graphStatus GetAttr(const AscendString &name, bool &attr_value) const;
  graphStatus GetAttr(const AscendString &name, Tensor &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<int64_t> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<int32_t> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<uint32_t> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<float> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<AscendString> &attr_values) const;
  graphStatus GetAttr(const AscendString &name, std::vector<bool> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<Tensor> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, OpBytes &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<std::vector<int64_t>> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, std::vector<ge::DataType> &attr_value) const;
  graphStatus GetAttr(const AscendString &name, ge::DataType &attr_value) const;
  graphStatus GetAttr(const AscendString &name, AttrValue &attr_value) const;

  graphStatus SetAttr(const AscendString &name, int64_t &attr_value) const;
  graphStatus SetAttr(const AscendString &name, int32_t &attr_value) const;
  graphStatus SetAttr(const AscendString &name, uint32_t &attr_value) const;
  graphStatus SetAttr(const AscendString &name, float &attr_value) const;
  graphStatus SetAttr(const AscendString &name, AscendString &attr_value) const;
  graphStatus SetAttr(const AscendString &name, bool &attr_value) const;
  graphStatus SetAttr(const AscendString &name, Tensor &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<int64_t> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<int32_t> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<uint32_t> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<float> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<AscendString> &attr_values) const;
  graphStatus SetAttr(const AscendString &name, std::vector<bool> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<Tensor> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, OpBytes &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<std::vector<int64_t>> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, std::vector<ge::DataType> &attr_value) const;
  graphStatus SetAttr(const AscendString &name, ge::DataType &attr_value) const;
  graphStatus SetAttr(const AscendString &name, AttrValue &attr_value) const;

  bool HasAttr(const AscendString &name);

  graphStatus GetSubgraph(uint32_t index, GraphPtr &graph) const;

  graphStatus GetALLSubgraphs(std::vector<GraphPtr> &graph_list) const;

 private:
   std::shared_ptr<NodeImpl> impl_;
   friend class NodeAdapter;
};
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_NODE_H_
