#include "optimize/pass/pass.h"

namespace fxrt {
namespace pass {
inline ir::NodePtr NodePass::NewTensor(ops::Op op, const std::vector<ir::NodePtr>& inputs) {
  return PassManager::Instance().NewTensor(op, inputs);
}

void PassManager::Run(ir::GraphPtr graph, const TensorCreator& creator) {
  if (passes_.empty() || graph->nodes.size() == 0) {
    RT_VLOG(VL_OPTIMIZE) << "No pass or no node in graph. instance: " << this;
    return;
  }
  RT_VLOG(VL_OPTIMIZE) << "Start running passes.";
  tensorCreator_ = creator;
  // Initialize the tensor list and map.
  orderedNodes_ = OrderedNodes(graph);
  ud_ = UserDef(graph);

  // Do transform for each node.
  auto tensors = orderedNodes_.tensorList();
  for (const ir::NodePtr& node : tensors) {
    RT_VLOG(VL_OPTIMIZE) << "Handle node: " << node;
    for (auto& pass : passes_) {
      RT_VLOG(VL_OPTIMIZE) << "Handle pass '" << pass.first << "'";
      if (!pass.second->Match(node)) {
        continue;
      }
      RT_VLOG(VL_OPTIMIZE) << "Matched pass '" << pass.first << "'";
      auto newNode = pass.second->Replacement();
      CHECK_IF_NULL(newNode);
      CHECK_IF_FAIL(Replace(node, newNode));
    }
  }

  // Flush optimized ordered nodes into graph.
  orderedNodes_.Flush(graph);
}

bool PassManager::Replace(const ir::NodePtr oldNode, const ir::NodePtr newNode) {
  RT_VLOG(VL_OPTIMIZE) << "To replace " << oldNode << " to " << newNode
                       << ", nodes size: " << orderedNodes_.tensorList().size();
  auto users = ud_.FindUsers(oldNode);
  if (users.empty()) {
    RT_GLOG(ERROR) << "No user for node: " << oldNode;
    return false;
  }
  RT_VLOG(VL_OPTIMIZE) << "users size: " << users.size();

  // Replace old node with new node.
  // We add node firstly, then remove old node.
  for (auto& user : users) {
    ir::NodePtr owner = user.first;
    InsertOrderedNodes(owner, user.second, oldNode, newNode);
    RemoveOrderedNodes(owner, user.second, oldNode);
    owner->inputs[user.second] = newNode;
  }
  RT_VLOG(VL_OPTIMIZE) << "Finish replace, nodes size: " << orderedNodes_.tensorList().size();

  // Since we run add&delete, now we can free unused tensor safely.
  for (auto& unused : unusedList_) {
    // Fix later: free all unused tensors.
    (void)unused;
  }
  return true;
}

void PassManager::RemoveOrderedNodes(const ir::NodePtr owner, size_t index, const ir::NodePtr node) {
  if (!ud_.DropNode(owner, index, node)) { // 'node' has other users.
    RT_VLOG(VL_OPTIMIZE) << "Has other users, " << node;
    return;
  }
  // 'node' has no user anymore.
  RT_VLOG(VL_OPTIMIZE) << "Run real remove, " << node;
  (void)unusedList_.emplace_back(node);
  if (!orderedNodes_.Remove(node)) {
    return;
  }
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    RemoveOrderedNodes(node, i, node->inputs[i]);
  }
}

void PassManager::InsertOrderedNodes(
    const ir::NodePtr owner,
    size_t index,
    const ir::NodePtr anchor,
    const ir::NodePtr node) {
  if (!ud_.AddNode(owner, index, node)) { // 'node' is not first insertion.
    RT_VLOG(VL_OPTIMIZE) << "Has other users, " << node;
    return;
  }
  if (!orderedNodes_.Insert(anchor, node)) {
    RT_VLOG(VL_OPTIMIZE) << "Insert failed, " << node;
    return;
  }
  RT_VLOG(VL_OPTIMIZE) << "Insert for inputs, " << node;
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    InsertOrderedNodes(node, i, anchor, node->inputs[i]);
  }
}

bool PassManager::OrderedNodes::Insert(const ir::NodePtr anchor, const ir::NodePtr node) {
  if (tensorMap_.count(node) != 0) {
    return false; // Already exists.
  }
  auto iter = tensorMap_.find(anchor);
  if (iter == tensorMap_.cend()) {
    return false; // Not found anchor.
  }
  auto listIter = tensorList_.emplace(iter->second, node);
  (void)tensorMap_.emplace(node, listIter);
  return true;
}

bool PassManager::OrderedNodes::Append(const ir::NodePtr node) {
  if (tensorMap_.find(node) != tensorMap_.cend()) {
    return false;
  }
  auto iter = tensorList_.emplace(tensorList_.end(), node);
  (void)tensorMap_.emplace(node, iter);
  return true;
}

bool PassManager::OrderedNodes::Remove(const ir::NodePtr node) {
  auto iter = tensorMap_.find(node);
  if (iter == tensorMap_.cend()) {
    return false;
  }
  tensorList_.erase(iter->second);
  tensorMap_.erase(iter);
  return true;
}

void PassManager::OrderedNodes::Init(ir::GraphPtr graph) {
  for (ssize_t i = graph->nodes.size() - 1; i >= 0; --i) {
    auto tensor = graph->nodes[i];
    auto iter = tensorList_.emplace(tensorList_.end(), tensor);
    (void)tensorMap_.emplace(tensor, iter);
  }
}

void PassManager::OrderedNodes::Flush(ir::GraphPtr graph) {
  graph->nodes.clear();
  for (auto iter = tensorList_.crbegin(); iter != tensorList_.crend(); ++iter) {
    (void)graph->nodes.emplace_back(*iter);
  }
}

class ManualSamplePass : public NodePass {
 public:
  ManualSamplePass() {}
  ~ManualSamplePass() = default;

  // If node is matched.
  bool Match(const ir::NodePtr node) override {
    node_ = node;
    RT_VLOG(VL_OPTIMIZE) << "To match " << node;
    return node->op == ops::Op_add;
  };

  // Replacement node for the matched node.
  ir::NodePtr Replacement() override {
    return NewTensor(ops::Op_mul, node_->inputs);
  };

 private:
  ir::NodePtr node_{nullptr};
};

DA_REGISTER_PASS("ManualSamplePass", ManualSamplePass);
} // namespace pass
} // namespace fxrt
