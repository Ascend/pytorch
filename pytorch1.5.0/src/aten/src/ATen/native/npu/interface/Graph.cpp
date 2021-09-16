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

#include <third_party/acl/inc/ge/ge_ir_build.h> // aclgrphGenerateForOp
#include "ATen/native/npu/interface/Graph.h"
#include <ATen/native/npu/interface/GeHelper.h>
#include <ATen/native/npu/utils/CalcuOpUtil.h>

namespace at {
namespace native {
namespace npu {

void GNodeAttrMaker::Set(ge::GNode& op, const ge::AscendString& name, bool value) {
  op.SetAttr(name, value);
}

void GNodeAttrMaker::Set(ge::GNode& op, const ge::AscendString& name, int64_t value) {
  op.SetAttr(name, value);
}

void GNodeAttrMaker::Set(ge::GNode& op, const ge::AscendString& name, float value) {
  op.SetAttr(name, value);
}

void GNodeAttrMaker::Set(ge::GNode& op, const ge::AscendString& name, std::string value) {
  ge::AscendString val(value.c_str());
  op.SetAttr(name, val);
}

void GNodeAttrMaker::Set(ge::GNode& op, const ge::AscendString& name, IntArrayRef value) {
  auto vec = value.vec();
  op.SetAttr(name, vec);
}

void GNodeAttrMaker::Set(ge::GNode& op, const ge::AscendString& name, at::ArrayRef<float> value) {
  auto vec = value.vec();
  op.SetAttr(name, vec);
}
void GNodeAttrMaker::Set(ge::GNode& op, const ge::AscendString& name, Scalar value){
  float val = CalcuOpUtil::get_scalar_float_value(value);
  op.SetAttr(name, val);
}

void GNodeAttrMaker::Set(ge::GNode& op, const ge::AscendString& name, at::ArrayRef<IntArrayRef> value){
  std::vector<std::vector<int64_t>> vals;
  for (int i = 0; i < value.size(); i++) {
    std::vector<int64_t> val;
    val.resize(value[i].size());
    std::copy(value[i].begin(), value[i].end(), val.begin());
    vals.emplace_back(val);
  }
  op.SetAttr(name, vals);
}


Graph& Graph::Name(const std::string& name) {
  this->name = name;
  this->inputs.clear();
  this->outputs.clear();
  return *this;
}

Graph& Graph::Input(const aclTensorDesc* inDesc) {
  inputs.emplace_back(GeHelper::Convert(inDesc));
  return *this;
}

Graph& Graph::Output(const aclTensorDesc* outDesc) {
  outputs.emplace_back(GeHelper::Convert(outDesc));
  return *this;
}

Graph& Graph::SetConst(void* const_data_buffer, const size_t &const_data_len) {
  TORCH_CHECK(inputs.size()>0, "The input vector can not be null!");
  // SetConstData function only support in CANN 5.0.3 (after 2021/08/15)
  // std::shared_ptr<void> data_ptr((void*) const_data_buffer, [](void*) {;});
  // inputs.back().SetConstData(data_ptr, const_data_len);
  return *this;
}

void Graph::Make() {
  if (not env::AutoTuneEnabled()) {
    return;
  }

  ge::AscendString opType(name.c_str());
  auto ret = ge::aclgrphGenerateForOp(opType, this->inputs, this->outputs, this->graph);
  if (ret != ge::GRAPH_SUCCESS) {
    AT_ERROR("aclgrphGenerateForOp failed. error code:", ret);
    return;
  }

  auto nodes = this->graph.GetDirectNode();
  ge::AscendString type;
  for (auto tmpNode: nodes) {
    tmpNode.GetType(type);
    if (type == name.c_str()) {
      node = tmpNode;
      break;
    }
  }

}

void Graph::GeGraph(ge::Graph& g) {
  g = this->graph;
}

} // namespace npu
} // namespace native
} // namespace at
