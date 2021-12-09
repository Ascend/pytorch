// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <c10/npu/NPUAny.h>
#include <c10/npu/NPUHashUtils.h>
#include <c10/util/Optional.h>
#include <third_party/acl/inc/graph/operator.h>

#include <functional>
#include <memory>
#include <vector>

namespace c10 {
namespace npu {
namespace graph {

constexpr uint32_t kDefaultMaxInputNum = 8;

// Node is the base class of npu graph
// It represents one computation in graph
class Node;

// A Value represents an input or output to node
// that is either a Tensor or an opaque Handle object
class Value;

using NodePtr = std::shared_ptr<Node>;
using ValueIndex = uint32_t;
using c10::npu::hash_utils::hash_t;
using DyNumAndIndex = std::vector<std::pair<uint32_t, uint32_t>>;
using DynamicInputRegFunc =
    std::function<ge::OperatorPtr(DyNumAndIndex, std::string)>;

class Value {
public:
  Value() = default;

  Value(NodePtr node, ValueIndex index)
      : cur_node_(node), value_index_(index) {}

  Value(NodePtr data, NodePtr node, ValueIndex index)
      : cur_node_(node), value_index_(index), data_node_(data) {}

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

  void UpdateFromOther(const Value& other) {
    if (other.data_node_.has_value()) {
      data_node_ = other.data_node_;
    }
    cur_node_ = other.cur_node_;
    value_index_ = other.value_index_;
    real_type_ = other.real_type_;
    value_hash_ = other.value_hash_;
  }

  void ResetValue() {
    cur_node_ = nullptr;
    value_index_ = 0;
    value_hash_ = c10::nullopt;
    data_node_ = c10::nullopt;
    real_type_ = c10::nullopt;
  }

private:
  NodePtr cur_node_ = nullptr;
  ValueIndex value_index_ = 0;
  c10::optional<NodePtr> data_node_ = c10::nullopt;
  c10::optional<hash_t> value_hash_ = c10::nullopt;
  c10::optional<std::string> real_type_ = c10::nullopt;
};

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

  const SmallVector<NodeInput, kDefaultMaxInputNum>& GetInputs() const {
    return inputs_;
  }

  void AddExtInfo(NodeExtInfoType ext_info_type, any any_attr) {
    ext_info_.emplace_back(ext_info_type, std::move(any_attr));
  }

  SmallVector<std::pair<NodeExtInfoType, any>, kDefaultMaxInputNum>&
  GetExtInfo() {
    return ext_info_;
  }

private:
  std::string op_type_;
  std::shared_ptr<ge::Operator> ge_op_ = nullptr;
  hash_t node_hash_ = hash_utils::hash_seed;
  SmallVector<NodeInput, kDefaultMaxInputNum> inputs_;
  SmallVector<std::pair<NodeExtInfoType, any>, kDefaultMaxInputNum> ext_info_;
};

} // namespace graph
} // namespace npu
} // namespace c10