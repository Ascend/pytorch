#ifndef INC_EXTERNAL_GRAPH_GRAPH_H_
#define INC_EXTERNAL_GRAPH_GRAPH_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "./operator.h"
#include "./gnode.h"

namespace ge {
class Graph;
class GraphImpl;

using GraphImplPtr = std::shared_ptr<GraphImpl>;
using GraphPtr = std::shared_ptr<Graph>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Graph {
  friend class GraphUtils;

 public:
  ATTRIBUTED_DEPRECATED(Graph(const char *))
  explicit Graph(const std::string &name);

  explicit Graph(const char *name);

  Graph() = default;

  ~Graph() = default;

  Graph &SetInputs(const std::vector<Operator> &inputs);

  Graph &SetOutputs(const std::vector<Operator> &outputs);

  Graph &SetOutputs(const std::vector<std::pair<Operator, std::vector<size_t>>> &output_indexs);

  ATTRIBUTED_DEPRECATED(Graph &SetOutputs(const std::vector<std::pair<ge::Operator, AscendString) &)
  Graph &SetOutputs(const std::vector<std::pair<ge::Operator, std::string>> &outputs);

  Graph &SetOutputs(const std::vector<std::pair<ge::Operator, AscendString>> &outputs);

  Graph &SetTargets(const std::vector<Operator> &targets);

  bool IsValid() const;

  graphStatus AddOp(const ge::Operator &op);

  ATTRIBUTED_DEPRECATED(graphStatus FindOpByName(const char *, ge::Operator &))
  graphStatus FindOpByName(const std::string &name, ge::Operator &op) const;

  graphStatus FindOpByName(const char *name, ge::Operator &op) const;

  ATTRIBUTED_DEPRECATED(graphStatus FindOpByType(const char *, std::vector<ge::Operator> &))
  graphStatus FindOpByType(const std::string &type, std::vector<ge::Operator> &ops) const;

  graphStatus FindOpByType(const char *type, std::vector<ge::Operator> &ops) const;

  ATTRIBUTED_DEPRECATED(graphStatus GetAllOpName(std::vector<AscendString> &) const)
  graphStatus GetAllOpName(std::vector<std::string> &op_name) const;

  graphStatus GetAllOpName(std::vector<AscendString> &names) const;

  ATTRIBUTED_DEPRECATED(graphStatus SaveToFile(const char *file_name) const)
  graphStatus SaveToFile(const std::string &file_name) const;

  graphStatus SaveToFile(const char *file_name) const;

  ATTRIBUTED_DEPRECATED(graphStatus LoadFromFile(const char *))
  graphStatus LoadFromFile(const std::string &file_name);

  graphStatus LoadFromFile(const char *file_name);

  ATTRIBUTED_DEPRECATED(graphStatus GetName(AscendString &) const)
  const std::string &GetName() const;

  graphStatus GetName(AscendString &name) const;

  ///
  /// Set is need train iteration.
  /// If set true, it means this graph need to be run iteration some
  /// times(according variant "npu_runconfig/iterations_per_loop").
  /// @param need_iteration need_iteration:whether to set iteration or not
  ///
  void SetNeedIteration(bool need_iteration);

  std::vector<GNode> GetAllNodes() const;

  std::vector<GNode> GetDirectNode () const;

  graphStatus RemoveNode(GNode &node);

  graphStatus RemoveNode(GNode &node, bool contain_subgraph);

  graphStatus RemoveEdge(GNode &src_node, const int32_t src_port_index, GNode &dst_node, const int32_t dst_port_index);

  GNode AddNodeByOp(const Operator &op);

  graphStatus AddDataEdge(GNode &src_node, const int32_t src_port_index,
                          GNode &dst_node, const int32_t dst_port_index);

  graphStatus AddControlEdge(GNode &src_node, GNode &dst_node);

  graphStatus CopyFrom(const Graph &src_graph);

  static GraphPtr ConstructFromInputs(const std::vector<Operator> &inputs, const AscendString &name);

 private:

  GraphImplPtr impl_{nullptr};
};
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_GRAPH_H_
