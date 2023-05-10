// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License")
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
#include "graph/operator.h"

namespace ge {

Operator op;

Operator::Operator(const char* type){};

Operator::Operator(const char* name, const char* type) {}

Operator::Operator(const AscendString& name, const AscendString& type) {}

void Operator::AttrRegister(
    const char* name,
    const std::vector<int64_t>& attr_value) {}

void Operator::AttrRegister(const char* name, const AscendString& attr_value) {}

void Operator::AttrRegister(const char* name, const Tensor& attr_value) {}

void Operator::AttrRegister(const char* name, bool attr_value) {}

void Operator::AttrRegister(const char* name, const ge::DataType& attr_value) {}

void Operator::AttrRegister(const char* name, float attr_value) {}

void Operator::AttrRegister(const char* name, int64_t attr_value) {}

void Operator::InputRegister(const char* name) {}

void Operator::DynamicInputRegisterByIndex(
    const char* name,
    const unsigned int num,
    size_t index) {}

void Operator::DynamicInputRegister(
    const char* name,
    const unsigned int num,
    bool is_push_back) {}

void Operator::DynamicOutputRegister(
    const char* name,
    const unsigned int num,
    bool is_push_back) {}

void Operator::RequiredAttrRegister(const char* name) {}

void Operator::OutputRegister(const char* name) {}

Operator& Operator::SetAttr(const char* name, const AscendString& attr_value) {}

graphStatus Operator::UpdateInputDesc(
    const char* name,
    const TensorDesc& tensor_desc) {
  return GRAPH_SUCCESS;
}

graphStatus Operator::UpdateOutputDesc(
    const char* name,
    const TensorDesc& tensor_desc) {
  return GRAPH_SUCCESS;
}

TensorDesc Operator::GetOutputDescByName(const char* name) const {
  return TensorDesc();
}

Operator& Operator::SetInput(
    uint32_t dst_index,
    const Operator& src_oprt,
    uint32_t src_index) {
  return op;
}

TensorDesc Operator::GetInputDescByName(const char* name) const {
  return TensorDesc();
}

Operator& Operator::SetAttr(const char* name, bool attr_value) {
  return op;
}

Operator& Operator::SetAttr(const char* name, int64_t attr_value) {
  return op;
}

Operator& Operator::SetAttr(const char* name, int32_t attr_value) {
  return op;
}

Operator& Operator::SetAttr(const char* name, uint32_t attr_value) {
  return op;
}

Operator& Operator::SetAttr(
    const char* name,
    const std::vector<int64_t>& attr_value) {
  return op;
}

Operator& Operator::SetAttr(
    const char* name,
    const std::vector<int32_t>& attr_value) {
  return op;
}

Operator& Operator::SetAttr(
    const char* name,
    const std::vector<uint32_t>& attr_value) {
  return op;
}

Operator& Operator::SetAttr(
    const char* name,
    std::initializer_list<int64_t>&& attr_value) {
  return op;
}

Operator& Operator::SetAttr(const char* name, float attr_value) {
  return op;
}

Operator& Operator::SetAttr(
    const char* name,
    const std::vector<float>& attr_value) {
  return op;
}

Operator& Operator::SetAttr(
    const char* name,
    const std::vector<bool>& attr_value) {
  return op;
}

Operator& Operator::SetAttr(const char* name, AttrValue&& attr_value) {
  return op;
}

Operator& Operator::SetAttr(const char* name, const Tensor& attr_value) {
  return op;
}

Operator& Operator::SetAttr(const char* name, const ge::DataType& attr_value){
  return op;
}

Operator& Operator::AddControlInput(const Operator& src_oprt) {
  return op;
}

} // namespace ge

