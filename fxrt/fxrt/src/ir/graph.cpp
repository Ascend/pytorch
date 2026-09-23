#include <unordered_map>

#include "ir/graph.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "ir/symbolic/symbolic.h"

namespace fxrt {
namespace ir {
namespace {

using NodeMap = std::unordered_map<Node*, NodePtr>;

bool IsEndNode(const NodePtr& node) {
  return node && node->op == ops::Op_End;
}

NodePtr CloneNodeShell(const NodePtr& old_node) {
  CHECK_IF_NULL(old_node);
  auto new_node = MakeIntrusive<Node>();
  new_node->op = old_node->op;
  new_node->attrs.reserve(old_node->attrs.size());
  for (const auto& attr : old_node->attrs) {
    if (attr.second != nullptr) {
      new_node->attrs.emplace(attr.first, attr.second->DeepCopy());
    }
  }
  if (old_node->output) {
    new_node->output = old_node->output->DeepCopy();
  }
  return new_node;
}

void CloneRegularNodes(const std::vector<NodePtr>& nodes, NodeMap* node_map) {
  CHECK_IF_NULL(node_map);
  for (const auto& old_node : nodes) {
    if (!old_node || IsEndNode(old_node)) {
      continue;
    }
    (*node_map)[old_node.get()] = CloneNodeShell(old_node);
  }
}

void RebuildInputs(const NodePtr& old_node, const NodeMap& node_map, const NodePtr& new_node) {
  CHECK_IF_NULL(old_node);
  CHECK_IF_NULL(new_node);
  for (const auto& old_input : old_node->inputs) {
    if (!old_input) {
      continue;
    }
    if (IsEndNode(old_input)) {
      new_node->inputs.push_back(old_input);
      continue;
    }
    auto input_it = node_map.find(old_input.get());
    if (input_it != node_map.end()) {
      new_node->inputs.push_back(input_it->second);
    }
  }
}

void RefreshSpecialNodeOutput(const NodePtr& new_node) {
  CHECK_IF_NULL(new_node);
  if (new_node->op == ops::Op_make_tuple) {
    std::vector<ValuePtr> input_values;
    input_values.reserve(new_node->inputs.size());
    for (const auto& input_node : new_node->inputs) {
      CHECK_IF_NULL(input_node);
      CHECK_IF_NULL(input_node->output);
      input_values.push_back(input_node->output);
    }
    new_node->output = MakeIntrusive<Value>(MakeIntrusive<Tuple>(std::move(input_values)));
    return;
  }

  if (new_node->op == ops::Op_tuple_getitem) {
    CHECK_IF_FAIL(new_node->inputs.size() == 2);
    auto tuple_node = new_node->inputs[0];
    auto index_node = new_node->inputs[1];
    CHECK_IF_NULL(tuple_node);
    CHECK_IF_NULL(index_node);
    CHECK_IF_FAIL(tuple_node->output && tuple_node->output->IsTuple());
    CHECK_IF_FAIL(index_node->output && index_node->output->IsInt());

    auto tuple_value = tuple_node->output->ToTuple();
    CHECK_IF_NULL(tuple_value);
    int64_t index = index_node->output->ToInt();
    CHECK_IF_FAIL(index >= 0 && index < static_cast<int64_t>(tuple_value->Size()));
    new_node->output = (*tuple_value)[index];
    return;
  }

  if (new_node->op == ops::Op_return) {
    CHECK_IF_FAIL(new_node->inputs.size() == 1);
    new_node->output = new_node->inputs[0]->output;
  }
}

void RebuildConnections(const std::vector<NodePtr>& nodes, const NodeMap& node_map) {
  for (const auto& old_node : nodes) {
    if (!old_node || IsEndNode(old_node)) {
      continue;
    }
    auto node_it = node_map.find(old_node.get());
    CHECK_IF_FAIL(node_it != node_map.end());
    auto& new_node = node_it->second;
    RebuildInputs(old_node, node_map, new_node);
    RefreshSpecialNodeOutput(new_node);
  }
}

void PopulateGraphNodes(const Graph& old_graph, const NodeMap& node_map, Graph* new_graph) {
  CHECK_IF_NULL(new_graph);
  new_graph->nodes.reserve(old_graph.nodes.size());
  for (const auto& old_node : old_graph.nodes) {
    if (!old_node) {
      continue;
    }
    if (IsEndNode(old_node)) {
      new_graph->nodes.push_back(old_node);
      continue;
    }
    auto node_it = node_map.find(old_node.get());
    if (node_it != node_map.end()) {
      new_graph->nodes.push_back(node_it->second);
    }
  }
}

} // namespace

GraphPtr Graph::DeepCopy() const {
  NodeMap node_map;
  CloneRegularNodes(nodes, &node_map);
  RebuildConnections(nodes, node_map);

  auto new_graph = MakeIntrusive<Graph>();
  // Preserve node ordering while reusing terminal nodes by design.
  PopulateGraphNodes(*this, node_map, new_graph.get());
  new_graph->parameters = parameters;
  new_graph->inputs = inputs;
  return new_graph;
}

} // namespace ir
} // namespace fxrt
