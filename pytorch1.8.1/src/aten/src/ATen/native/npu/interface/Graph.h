// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#ifndef __NATIVE_NPU_INTERFACE_GRAPH__
#define __NATIVE_NPU_INTERFACE_GRAPH__

#include <third_party/acl/inc/graph/graph.h> // ge::Graph
#include <third_party/acl/inc/graph/tensor.h> // TensorDesc
#include <third_party/acl/inc/graph/types.h> // Format
#include <third_party/acl/inc/acl/acl_base.h>
#include <ATen/native/npu/interface/EnvVariables.h>
#include "c10/util/Exception.h"
#include "ATen/ATen.h"
#include <iostream>

namespace at {
namespace native {
namespace npu {

/**
  This class is used to set GNode's attribute.
  */
class GNodeAttrMaker {
public:
  static void Set(ge::GNode& op, const ge::AscendString& name, bool value);
  static void Set(ge::GNode& op, const ge::AscendString& name, int64_t value);
  static void Set(ge::GNode& op, const ge::AscendString& name, float value);
  static void Set(ge::GNode& op, const ge::AscendString& name, std::string value);
  static void Set(ge::GNode& op, const ge::AscendString& name, IntArrayRef value);
  static void Set(ge::GNode& op, const ge::AscendString& name, at::ArrayRef<float> value);
  static void Set(ge::GNode& op, const ge::AscendString& name, Scalar value);
  static void Set(ge::GNode& op, const ge::AscendString& name, at::ArrayRef<IntArrayRef> value);
}; // class GNodeAttrMaker

/**
  Class Graph is the wrapper of ge::Graph andd support to use ACL interface to construct.
  */
class Graph {
public:
  /**
    This api is used to set graph's name.
    */
  Graph& Name(std::string name);
  /**
    This api is used to set graph's input desc
    */
  Graph& Input(const aclTensorDesc* inDesc);
  /**
    This api is used to set graph's output desc
    */
  Graph& Output(const aclTensorDesc* outDesc);
  /**
  This api is used to set graph's last input desc to be const.
  */
  Graph& SetConst(void* const_data_buffer, const size_t &const_data_len);
  /**
    This api is used to make graph, which is depend on the TensorDesc of inputs and outputs
    */
  void Make();
  /**
    This api should be called after Make().
    */
  template <typename dataType>
  void AddAttr(std::string& attrName, dataType value);
  /**
    This API is used to get the private member: ge::Graph.
    */
  void GeGraph(ge::Graph& g);

private:
  std::string name;
  std::vector<ge::TensorDesc> inputs;
  std::vector<ge::TensorDesc> outputs;
  ge::Graph graph;
  ge::GNode node;
};

template <typename dataType>
void Graph::AddAttr(std::string& attrName, dataType value) {
  if (not env::AutoTuneEnabled()) {
    return;
  }
  ge::AscendString attrName_(attrName.c_str());
  GNodeAttrMaker::Set(node, attrName_, value);
}
} // namespace npu
} // namespace native
} // namespace at

#endif // __NATIVE_NPU_INTERFACE_GRAPH__