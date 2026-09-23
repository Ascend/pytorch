#ifndef __IR_GRAPH_H__
#define __IR_GRAPH_H__

#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <unordered_map>

#include "ir/common/intrusive_ptr.h"

// Forward declarations to avoid circular dependencies
namespace fxrt {
namespace ir {
class Value;
using ValuePtr = IntrusivePtr<Value>;
class Graph;
using GraphPtr = IntrusivePtr<Graph>;
} // namespace ir
} // namespace fxrt

#include "ir/value/value.h"
#include "ops/op_def/ops_name.h"

namespace fxrt {
namespace ir {

/**
 * @brief Represents a node in the computation graph.
 *
 * A node corresponds to an operation, with a set of inputs and a single output.
 */
struct Node : public RefCounted {
  ops::Op op; ///< The operation performed by this node.
  std::vector<IntrusivePtr<Node>> inputs; ///< The input nodes to the operation.
  // Backend-neutral metadata attached by an importer (for example, compiled
  // kernel resource paths).  Values are deliberately typed IR values rather
  // than an ad-hoc string map so future backends can carry structured data.
  std::unordered_map<std::string, ValuePtr> attrs;
  ValuePtr output{nullptr}; ///< The output value from the operation.
};
using NodePtr = IntrusivePtr<Node>;

/**
 * @brief Represents the entire computation graph.
 */
struct Graph : public RefCounted {
  std::vector<IntrusivePtr<Node>> nodes; ///< The list of all value nodes and op nodes in the graph.
  std::vector<IntrusivePtr<Node>> inputs;
  std::vector<IntrusivePtr<Node>> parameters;

  /**
   * @brief Creates a deep copy of the graph, preserving node connections.
   * @return A new Graph object with copied nodes and preserved connection relationships.
   *
   * Special handling:
   * - Nodes with op == Op_End are shallow copied (shared between graphs)
   * - make_tuple and tuple_getitem operations maintain value reference semantics
   * - parameters list is shallow copied
   * - For Tensor values, new Storage objects own their data (ownsData_ = true)
   */
  GraphPtr DeepCopy() const;

  // void Dump();
  // std::unordered_map<ir::NodePtr, size_t> paraNumMap_;
  // std::unordered_map<ir::NodePtr, size_t> nodeNumMap_;
};

using GraphPtr = IntrusivePtr<Graph>;

inline std::ostream& operator<<(std::ostream& os, const Node& node) {
  os << "Node("
     << "op=" << ops::ToStr(node.op) << ", value=" << node.output << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const NodePtr& node) {
  if (node == nullptr) {
    os << "Null";
  } else {
    os << *node;
  }
  return os;
}

} // namespace ir
} // namespace fxrt

#endif // __IR_GRAPH_H__
