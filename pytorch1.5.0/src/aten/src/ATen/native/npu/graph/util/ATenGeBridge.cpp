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

#include "ATenGeBridge.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include <third_party/acl/inc/graph/operator_factory.h>
#include <third_party/acl/inc/op_proto/array_ops.h>

namespace at {
namespace native {
namespace npu {
namespace {
std::map<at::ScalarType, ge::DataType> kScalarTypeToGeDType{
    {at::ScalarType::Byte, ge::DataType::DT_UINT8},
    {at::ScalarType::Char, ge::DataType::DT_INT8},
    {at::ScalarType::Bool, ge::DataType::DT_BOOL},
    {at::ScalarType::Double, ge::DataType::DT_DOUBLE},
    {at::ScalarType::Float, ge::DataType::DT_FLOAT},
    {at::ScalarType::Half, ge::DataType::DT_FLOAT16},
    {at::ScalarType::Short, ge::DataType::DT_INT16},
    {at::ScalarType::Int, ge::DataType::DT_INT32},
    {at::ScalarType::Long, ge::DataType::DT_INT64},
};

at::Tensor ConstructCpuTenosr(const Scalar& scalar_input, ScalarType type) {
  return scalar_to_tensor(scalar_input).to(type);
}

at::Tensor ConstructCpuTenosr(
    const std::vector<int64_t>& list_input,
    ScalarType dtype) {
  auto cpu_tensor = at::from_blob(
      const_cast<void*>(reinterpret_cast<const void*>(list_input.data())),
      {list_input.size()},
      TensorOptions(kCPU).dtype(at::kLong));
  if (dtype != at::kLong) {
    return cpu_tensor.to(dtype);
  }
  return cpu_tensor;
}
} // namespace

ge::DataType ATenGeBridge::GetGeDType(ScalarType type) {
  auto iter = kScalarTypeToGeDType.find(type);
  if (iter == kScalarTypeToGeDType.end()) {
    AT_ERROR("Unsupported convert this ATen DType: %s to Ge DType", type);
  }
  return iter->second;
}
ge::DataType ATenGeBridge::GetGeDType(caffe2::TypeMeta type_meta) {
  auto aten_dtype = c10::typeMetaToScalarType(type_meta);
  return GetGeDType(aten_dtype);
}

ge::Shape ATenGeBridge::GetGeShape(ArrayRef<int64_t> vec) {
  return ge::Shape(std::vector<int64_t>(vec.begin(), vec.end()));
}

ge::TensorDesc ATenGeBridge::InferGeTenosrDesc(
    const NPUStorageDesc& storage_desc,
    const caffe2::TypeMeta& type_meta) {
  ge::TensorDesc tensor_desc;
  tensor_desc.SetDataType(ATenGeBridge::GetGeDType(type_meta));
  tensor_desc.SetPlacement(ge::kPlacementDevice);
  tensor_desc.SetOriginShape(
      ATenGeBridge::GetGeShape(storage_desc.base_sizes_));
  tensor_desc.SetShape(ATenGeBridge::GetGeShape(storage_desc.storage_sizes_));

  tensor_desc.SetOriginFormat(ge::Format(storage_desc.origin_format_));
  tensor_desc.SetFormat(ge::Format(storage_desc.npu_format_));

  return tensor_desc;
}

template <typename ConstType>
void ATenGeBridge::SetGeOpConstInput(
    const c10::any& const_input,
    ge::OperatorPtr ge_op) {
  auto const_input_tuple =
      ATenGeBridge::TryToGetAnyValue<ConstType>(const_input);
  at::Tensor cpu_tensor = ConstructCpuTenosr(
      std::get<1>(const_input_tuple), std::get<2>(const_input_tuple));
  auto ge_data_type = GetGeDType(std::get<2>(const_input_tuple));
  ge::TensorDesc ge_tensor_desc{
      ge::Shape(cpu_tensor.sizes().vec()), ge::Format::FORMAT_ND, ge_data_type};
  ge::Tensor ge_tenosr{
      ge_tensor_desc,
      reinterpret_cast<uint8_t*>(cpu_tensor.data_ptr()),
      cpu_tensor.nbytes()};

  auto const_op = std::make_shared<ge::op::Const>();
  const_op->set_attr_value(ge_tenosr);
  ge_op->SetInput(std::get<0>(const_input_tuple), *const_op, 0);
}

void ATenGeBridge::SetSensitiveFormat(
    const c10::any& sensitive_format,
    ge::OperatorPtr ge_op,
    NodeExtInfoType ext_type) {
  auto sensitive_format_pair =
      TryToGetAnyValue<std::pair<string, aclFormat>>(sensitive_format);
  if (ext_type == NodeExtInfoType::SENSITIVE_FORMAT_OF_INPUT) {
    auto tmp_desc =
        ge_op->GetInputDescByName(sensitive_format_pair.first.c_str());
    tmp_desc.SetFormat(ge::Format(sensitive_format_pair.second));
    tmp_desc.SetOriginFormat(ge::Format(sensitive_format_pair.second));
    ge_op->UpdateInputDesc(sensitive_format_pair.first, tmp_desc);
  } else {
    auto tmp_desc =
        ge_op->GetOutputDescByName(sensitive_format_pair.first.c_str());
    tmp_desc.SetFormat(ge::Format(sensitive_format_pair.second));
    tmp_desc.SetOriginFormat(ge::Format(sensitive_format_pair.second));
    ge_op->UpdateOutputDesc(sensitive_format_pair.first, tmp_desc);
  }
}

void ATenGeBridge::AddNodeExtInfoIntoGeOp(
    ArrayRef<std::pair<NodeExtInfoType, c10::any>> ext_info,
    ge::OperatorPtr ge_op) {
  for (const auto& info : ext_info) {
    switch (info.first) {
      case NodeExtInfoType::ATTR_TYPE_BOOL:
        SetGeOpAttr<std::pair<string, bool>>(info.second, ge_op);
        break;
      case NodeExtInfoType::ATTR_TYPE_LONG:
        SetGeOpAttr<std::pair<string, int64_t>>(info.second, ge_op);
        break;
      case NodeExtInfoType::ATTR_TYPE_FLOAT:
        SetGeOpAttr<std::pair<string, float>>(info.second, ge_op);
        break;
      case NodeExtInfoType::ATTR_TYPE_STRING:
        SetGeOpAttr<std::pair<string, string>>(info.second, ge_op);
      case NodeExtInfoType::ATTR_TYPE_LIST_LONG:
        SetGeOpAttr<std::pair<string, vector<int64_t>>>(info.second, ge_op);
        break;
      case NodeExtInfoType::ATTR_TYPE_LIST_FLOAT:
        AT_ERROR("Not Implemented");
        break;
      case NodeExtInfoType::INPUT_TYPE_SCALAR:
        SetGeOpConstInput<std::tuple<uint32_t, Scalar, ScalarType>>(
            info.second, ge_op);
        break;
      case NodeExtInfoType::INPUT_TYPE_LIST_LONG:
        SetGeOpConstInput<std::tuple<uint32_t, vector<int64_t>, ScalarType>>(
            info.second, ge_op);
        break;
      case NodeExtInfoType::SENSITIVE_FORMAT_OF_INPUT:
        SetSensitiveFormat(
            info.second, ge_op, NodeExtInfoType::SENSITIVE_FORMAT_OF_INPUT);
        break;
      case NodeExtInfoType::SENSITIVE_FORMAT_OF_OUTPUT:
        SetSensitiveFormat(
            info.second, ge_op, NodeExtInfoType::SENSITIVE_FORMAT_OF_OUTPUT);
        break;
      default:
        AT_ERROR(
            "Has no method to process node ext info type: %d",
            static_cast<std::underlying_type<NodeExtInfoType>::type>(
                info.first));
    }
  }
}

void ATenGeBridge::PorcessDynamicInputReg(
    NodePtr node,
    ge::OperatorPtr ge_op,
    string op_name) {
  auto& ext_info = node->GetExtInfo();
  auto it = std::find_if(
      ext_info.begin(),
      ext_info.end(),
      [](const std::pair<NodeExtInfoType, c10::any>& item) {
        return item.first == NodeExtInfoType::DYNAMIC_INPUT_FUNC;
      });
  if (it != ext_info.end()) {
    auto func_and_para =
        TryToGetAnyValue<std::pair<DynamicInputRegFunc, DyNumAndIndex>>(
            it->second);
    ge_op = func_and_para.first(func_and_para.second, op_name);

    // no need to process it anymore
    ext_info.erase(it);
  }
  return;
}

void ATenGeBridge::CheckAndBuildGeOpForNode(NodePtr node) {
  if (node->GetGeOp() != nullptr) {
    return;
  }
  static uint64_t op_index = 0;
  const string op_type = node->GetOpType();
  TORCH_CHECK(
      ge::OperatorFactory::IsExistOp(op_type),
      "Cur op type: %s is not exit",
      op_type);
  string op_name = op_type + std::to_string(op_index++);
  ge::OperatorPtr ge_op = nullptr;
  PorcessDynamicInputReg(node, ge_op, op_name);
  if (ge_op == nullptr) {
    ge_op = std::make_shared<ge::Operator>(
        ge::OperatorFactory::CreateOperator(op_name, op_type));
  }
  AddNodeExtInfoIntoGeOp(node->GetExtInfo(), ge_op);
  node->SetGeOp(ge_op);
  return;
}

} // namespace npu
} // namespace native
} // namespace at