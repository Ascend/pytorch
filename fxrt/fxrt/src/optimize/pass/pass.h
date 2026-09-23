#ifndef __OPTIMIZE_PASS_PASS_H__
#define __OPTIMIZE_PASS_PASS_H__

#include <utility>
#include <functional>
#include <iterator>
#include <list>
#include <string>
#include <vector>
#include <unordered_map>

#include "ir/graph.h"
#include "optimize/pass/ud.h"

namespace fxrt {
namespace pass {

using TensorCreator = std::function<ir::NodePtr(ops::Op, const std::vector<ir::NodePtr>&)>;

class NodePass {
 public:
  // If node is matched.
  virtual bool Match(const ir::NodePtr node) = 0;

  // Replacement node for the matched node.
  virtual ir::NodePtr Replacement() = 0;

  ir::NodePtr NewTensor(ops::Op op, const std::vector<ir::NodePtr>& inputs);
};

class PassManager {
 public:
  PassManager() = default;

  static PassManager& Instance() {
    static PassManager instance;
    RT_VLOG(VL_OPTIMIZE) << "PassManager instance: '" << &instance << "'";
    return instance;
  }

  void Run(ir::GraphPtr graph, const TensorCreator& creator);

  void AddPass(const std::string& name, const NodePass& pass) {
    RT_VLOG(VL_OPTIMIZE) << "Add pass '" << name << "'";
    (void)passes_.emplace_back(std::make_pair(name, const_cast<NodePass*>(&pass)));
  }

  ir::NodePtr NewTensor(ops::Op op, const std::vector<ir::NodePtr>& inputs) {
    CHECK_IF_FAIL(tensorCreator_);
    return tensorCreator_(op, inputs);
  }

 private:
  using TensorList = std::list<ir::NodePtr>;

  bool Replace(const ir::NodePtr oldNode, const ir::NodePtr newNode);
  void RemoveOrderedNodes(const ir::NodePtr owner, size_t index, const ir::NodePtr node);
  void InsertOrderedNodes(const ir::NodePtr owner, size_t index, const ir::NodePtr anchor, const ir::NodePtr node);

  class OrderedNodes {
   public:
    OrderedNodes() = default;
    explicit OrderedNodes(ir::GraphPtr graph) {
      Init(graph);
    }

    // Return true if inserted, or false otherwise.
    bool Insert(const ir::NodePtr anchor, const ir::NodePtr node);
    // Return true if appended, or false otherwise.
    bool Append(const ir::NodePtr node);
    // Return true if erased, or false otherwise.
    bool Remove(const ir::NodePtr node);
    // Build the ordered nodes.
    void Init(ir::GraphPtr graph);
    // Flush the nodes back into graph.
    void Flush(ir::GraphPtr graph);

    const TensorList& tensorList() const {
      return tensorList_;
    }

   private:
    TensorList tensorList_;
    std::unordered_map<ir::NodePtr, TensorList::iterator> tensorMap_;
  } orderedNodes_;

  std::vector<std::pair<std::string, NodePass*>> passes_;
  UserDef ud_;
  std::vector<ir::NodePtr> unusedList_;
  TensorCreator tensorCreator_;
};

class PassRegister {
 public:
  PassRegister(const std::string& name, const NodePass& pass) {
    RT_VLOG(VL_OPTIMIZE) << "Register pass '" << name;
    PassManager::Instance().AddPass(name, pass);
  }
};

#define DA_REGISTER_PASS(PASS_NAME, PASS) \
  static const PASS __pass_##PASS##__;    \
  static const PassRegister __passRegister_##PASS##__(PASS_NAME, __pass_##PASS##__);
} // namespace pass
} // namespace fxrt

#endif // __OPTIMIZE_PASS_PASS_H__
