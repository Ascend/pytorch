#pragma once

#include "torch_npu/csrc/framework/graph/util/NPUAny.h"
#include "torch_npu/csrc/framework/graph/util/NPUHashUtils.h"
#include <c10/util/Optional.h>
#include <third_party/acl/inc/graph/operator.h>

#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>

namespace at_npu {
namespace native {

using namespace at_npu::native::hash_utils;

constexpr uint32_t kDefaultMaxInputNum = 8;

// Node is the base class of npu graph
// It represents one computation in graph
class Node;

// A Value represents an input or output to node
// that is either a Tensor or an opaque Handle object
class Value;

using NodePtr = std::shared_ptr<Node>;
using NodeWeakPtr = std::weak_ptr<Node>;
using ValueIndex = uint32_t;
using hash_utils::hash_t;
using DyNumAndIndex = std::vector<std::pair<uint32_t, uint32_t>>;
using DynamicInputRegFunc =
    std::function<ge::OperatorPtr(DyNumAndIndex, std::string)>;

struct NodeInput {
  NodeInput() = default;
  NodeInput(ValueIndex in_index, ValueIndex peer_index, NodePtr peer_node)
      : input_index(in_index),
        peer_output_index(peer_index),
        peer_output_node(peer_node) {}

  ValueIndex input_index = 0;
  ValueIndex peer_output_index = 0;
  NodePtr peer_output_node = nullptr;
};

enum class NodeExtInfoType : uint8_t {
  ATTR_TYPE_BOOL = 1,
  ATTR_TYPE_LONG,
  ATTR_TYPE_FLOAT,
  ATTR_TYPE_STRING,
  ATTR_TYPE_DATATYPE,
  ATTR_TYPE_LIST_BOOL,
  ATTR_TYPE_LIST_LONG,
  ATTR_TYPE_LIST_FLOAT,
  INPUT_TYPE_SCALAR,
  INPUT_TYPE_LIST_LONG,
  SENSITIVE_FORMAT_OF_INPUT,
  SENSITIVE_FORMAT_OF_OUTPUT,
  DYNAMIC_INPUT_FUNC,

};

class Node {
public:
  explicit Node(std::string op_type) : op_type_(std::move(op_type)) {
    node_hash_ = hash_utils::multi_hash(node_hash_, op_type_);
  };

  ~Node() {};

  std::string GetOpType() const {
    return op_type_;
  }

  void SetOpType(std::string op_type) {
    op_type_ = std::move(op_type);
    node_hash_ = hash_utils::multi_hash(node_hash_, op_type_);
  }

  std::shared_ptr<ge::Operator> GetGeOp() {
    return ge_op_;
  }

  void SetGeOp(const std::shared_ptr<ge::Operator>& op) {
    ge_op_ = op;
  }

  void Reset() {
    op_type_.clear();
    ge_op_ = nullptr;
    node_hash_ = hash_utils::hash_seed;
    inputs_.clear();
    ext_info_.clear();
    is_inplace_ = false;
    if (inplace_info_.has_value()) {
      inplace_info_.value().clear();
    }
  }

  template <typename... Args>
  void UpdateNodeHash(Args&&... args) {
    node_hash_ =
        hash_utils::multi_hash(node_hash_, std::forward<Args>(args)...);
  }

  hash_t GetNodeHash() const {
    return node_hash_;
  }

  void AddInput(
      ValueIndex input_index,
      NodePtr output_node,
      ValueIndex output_index) {
    inputs_.emplace_back(input_index, output_index, output_node);
  }

  const c10::SmallVector<NodeInput, kDefaultMaxInputNum>& GetInputs() const {
    return inputs_;
  }

  void AddExtInfo(NodeExtInfoType ext_info_type, Any any_attr) {
    ext_info_.emplace_back(ext_info_type, std::move(any_attr));
  }

  c10::SmallVector<std::pair<NodeExtInfoType, Any>, kDefaultMaxInputNum>&
  GetExtInfo() {
    return ext_info_;
  }

  void SetNodeInplace(bool is_inplace) {
    is_inplace_ = is_inplace;
  }

  bool IsInplace() const {
    return is_inplace_;
  }

  void SetInplaceNode(ValueIndex output_index, NodeWeakPtr inplace_node) {
    auto node_ptr = inplace_node.lock();
    if (node_ptr != nullptr) {
      if (!inplace_info_.has_value()) {
        inplace_info_ = {std::pair<ValueIndex, NodeWeakPtr>(output_index, inplace_node)};
      } else {
        inplace_info_.value().insert(std::pair<ValueIndex, NodeWeakPtr>(output_index, inplace_node));
      }
      node_ptr->SetNodeInplace(true);
    }
  }

  c10::optional<NodeWeakPtr> GetInplaceNode(ValueIndex output_index) const {
    if (!inplace_info_.has_value()) {
      return c10::nullopt;
    }
    const auto& inplace_map = inplace_info_.value();
    const auto& iter = inplace_map.find(output_index);
    if (iter != inplace_map.end()) {
      return iter->second;
    }
    return c10::nullopt;
  }

private:
  std::string op_type_;
  std::shared_ptr<ge::Operator> ge_op_ = nullptr;
  hash_t node_hash_ = hash_utils::hash_seed;
  c10::SmallVector<NodeInput, kDefaultMaxInputNum> inputs_;
  c10::SmallVector<std::pair<NodeExtInfoType, Any>, kDefaultMaxInputNum> ext_info_;
  bool is_inplace_ = false;
  c10::optional<std::unordered_map<ValueIndex, NodeWeakPtr>> inplace_info_ = c10::nullopt;
};

class Value {
public:
  Value() = default;

  Value(NodePtr node, ValueIndex index)
      : cur_node_(node), value_index_(index) {}

  Value(NodePtr data, NodePtr node, ValueIndex index)
      : cur_node_(node), value_index_(index), data_node_(data) {}

  ~Value() = default;

  NodePtr GetCurNode() const {
    return cur_node_;
  }

  c10::optional<NodePtr> GetDataNode() {
    return data_node_;
  }

  const c10::optional<std::string>& GetRealDtype() const{
    return real_type_;
  }

  ValueIndex GetValueIndex() const {
    return value_index_;
  }

  bool HashNode() const {
    return cur_node_ != nullptr;
  }

  void SetRealType(const std::string& real_type) {
    real_type_ = real_type;
  }

  hash_t GetValueHash() const;

  void SetScalarMemOffset(uint32_t addr_offset) {
    scalar_mem_offset_ = addr_offset;
  }

  c10::optional<uint32_t> GetScalarMemOffset() const {
    return scalar_mem_offset_;
  }

  void UpdateFromOther(const Value& other) {
    if ((cur_node_ != nullptr) && (other.cur_node_ != nullptr)) {
      NodeWeakPtr inplace_node(other.cur_node_);
      cur_node_->SetInplaceNode(value_index_, inplace_node);
    }
    this->SetFromOther(other);
  }

  void SetFromOther(const Value& other) {
    if (other.data_node_.has_value()) {
      data_node_ = other.data_node_;
    }
    cur_node_ = other.cur_node_;
    value_index_ = other.value_index_;
    real_type_ = other.real_type_;
    value_hash_ = other.value_hash_;
    scalar_mem_offset_ = other.scalar_mem_offset_;
  }

  void ResetValue() {
    cur_node_ = nullptr;
    value_index_ = 0;
    value_hash_ = c10::nullopt;
    data_node_ = c10::nullopt;
    real_type_ = c10::nullopt;
    scalar_mem_offset_ = c10::nullopt;
  }

private:
  NodePtr cur_node_ = nullptr;
  ValueIndex value_index_ = 0;
  c10::optional<NodePtr> data_node_ = c10::nullopt;
  c10::optional<hash_t> value_hash_ = c10::nullopt;
  c10::optional<std::string> real_type_ = c10::nullopt;
  c10::optional<uint32_t> scalar_mem_offset_ = c10::nullopt;
};

} // namespace graph
} // namespace at_npu

