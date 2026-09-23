#include "ops/ascend/atb/atb_adapter.h"
#include <iostream>
#include <map>
#include "hardware/device.h"
#include "ops/utils/utils.h"

namespace fxrt {
namespace ops {

namespace {

aclDataType ConvertToAclDataType(ir::DataType::Type type) {
  static const std::map<ir::DataType::Type, aclDataType> kDataTypeToAclDataTypeMap = {
      {ir::DataType::Type::Unknown, ACL_DT_UNDEFINED},
      {ir::DataType::Type::Float16, ACL_FLOAT16},
      {ir::DataType::Type::BFloat16, ACL_BF16},
      {ir::DataType::Type::Float32, ACL_FLOAT},
      {ir::DataType::Type::Float64, ACL_DOUBLE},
      {ir::DataType::Type::Complex64, ACL_COMPLEX64},
      {ir::DataType::Type::Int8, ACL_INT8},
      {ir::DataType::Type::Int16, ACL_INT16},
      {ir::DataType::Type::Int32, ACL_INT32},
      {ir::DataType::Type::Int64, ACL_INT64},
      {ir::DataType::Type::UInt8, ACL_UINT8},
      {ir::DataType::Type::Bool, ACL_BOOL},
      {ir::DataType::Type::QInt8, ACL_DT_UNDEFINED},
      {ir::DataType::Type::QUInt4x2, ACL_DT_UNDEFINED},
  };

  auto it = kDataTypeToAclDataTypeMap.find(type);
  if (it != kDataTypeToAclDataTypeMap.end()) {
    return it->second;
  }

  RT_GLOG(EXCEPTION) << "Failed to convert data type " << static_cast<int>(type) << " to ACL data type";
  return ACL_DT_UNDEFINED;
}

aclFormat ConvertMemoryFormatToAclFormat(ir::MemoryFormat format) {
  static const std::map<ir::MemoryFormat, aclFormat> kMemoryFormatToAclFormatMap = {
      {ir::MemoryFormat::FORMAT_UNDEFINED, ACL_FORMAT_UNDEFINED},
      {ir::MemoryFormat::FORMAT_NCHW, ACL_FORMAT_NCHW},
      {ir::MemoryFormat::FORMAT_NHWC, ACL_FORMAT_NHWC},
      {ir::MemoryFormat::FORMAT_ND, ACL_FORMAT_ND},
      {ir::MemoryFormat::FORMAT_NC1HWC0, ACL_FORMAT_NC1HWC0},
      {ir::MemoryFormat::FORMAT_FRACTAL_Z, ACL_FORMAT_FRACTAL_Z},
      {ir::MemoryFormat::FORMAT_NC1HWC0_C04, ACL_FORMAT_NC1HWC0_C04},
      {ir::MemoryFormat::FORMAT_HWCN, ACL_FORMAT_HWCN},
      {ir::MemoryFormat::FORMAT_NDHWC, ACL_FORMAT_NDHWC},
      {ir::MemoryFormat::FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ},
      {ir::MemoryFormat::FORMAT_NCDHW, ACL_FORMAT_NCDHW},
      {ir::MemoryFormat::FORMAT_NDC1HWC0, ACL_FORMAT_NDC1HWC0},
      {ir::MemoryFormat::FORMAT_FRACTAL_Z_3D, ACL_FRACTAL_Z_3D},
      {ir::MemoryFormat::FORMAT_NC, ACL_FORMAT_NC},
      {ir::MemoryFormat::FORMAT_NCL, ACL_FORMAT_NCL},
  };

  auto iter = kMemoryFormatToAclFormatMap.find(format);
  if (iter == kMemoryFormatToAclFormatMap.end()) {
    RT_GLOG(EXCEPTION) << "Unsupported MemoryFormat " << format << " for conversion to aclFormat";
    return ACL_FORMAT_UNDEFINED;
  }

  return iter->second;
}

aclFormat GetFormatForAtb(ir::MemoryFormat format) {
  // For base layouts, normalize to ND so that ATB kernels do not
  // rely on specific 4D/5D layout enums when the tensor is logically ND.
  return ops::IsBaseFormat(format) ? ACL_FORMAT_ND : ConvertMemoryFormatToAclFormat(format);
}

atb::Tensor GetAtbTensor(const ir::Value* value) {
  if (value == nullptr) {
    return atb::Tensor{};
  }

  auto tensor = value->ToTensor();
  CHECK_IF_NULL(tensor);

  atb::Tensor atb_tensor;
  const auto& shape = tensor->Shape();
  const auto shape_size = shape.size();

  atb_tensor.desc.dtype = ConvertToAclDataType(tensor->Dtype());
  atb_tensor.desc.format = GetFormatForAtb(tensor->Format());
  atb_tensor.desc.shape.dimNum = shape_size;
  for (size_t i = 0; i < shape_size; i++) {
    atb_tensor.desc.shape.dims[i] = static_cast<int32_t>(shape[i]);
  }

  atb_tensor.dataSize = tensor->Numel() * tensor->Dtype().GetSize();

  void* data_ptr = tensor->DataPtr();
  auto device = tensor->GetDevice();
  if (device.type == hardware::DeviceType::CPU) {
    atb_tensor.hostData = data_ptr;
    atb_tensor.deviceData = nullptr;
  } else {
    atb_tensor.deviceData = data_ptr;
    atb_tensor.hostData = nullptr;
  }

  if (!tensor->IsContiguous()) {
    RT_GLOG(EXCEPTION) << "Only contiguous tensor is supported in atb now.";
  }

  return atb_tensor;
}

void UpdateAddress(ir::Tensor* tensor, atb::Tensor* atb_tensor) {
  CHECK_IF_NULL(tensor);
  CHECK_IF_NULL(atb_tensor);

  void* data_ptr = tensor->DataPtr();
  auto device = tensor->GetDevice();
  if (device.type == hardware::DeviceType::CPU) {
    atb_tensor->hostData = data_ptr;
    atb_tensor->deviceData = nullptr;
  } else {
    atb_tensor->deviceData = data_ptr;
    atb_tensor->hostData = nullptr;
  }
}

} // namespace

ParamSetter& ParamSetter::Input(const ir::Value* value) {
  atb::Tensor tensor = GetAtbTensor(value);
  variant_pack.inTensors.push_back(std::move(tensor));
  return *this;
}

ParamSetter& ParamSetter::Input(std::optional<const ir::Value*> value) {
  if (value.has_value()) {
    return Input(value.value());
  }
  return Input(nullptr);
}

ParamSetter& ParamSetter::Output(const ir::Value* value) {
  atb::Tensor tensor = GetAtbTensor(value);
  variant_pack.outTensors.push_back(std::move(tensor));
  return *this;
}

ParamSetter& ParamSetter::Output(std::optional<const ir::Value*> value) {
  if (value.has_value()) {
    return Output(value.value());
  }
  return Output(nullptr);
}

void ParamSetter::Update(const std::vector<const ir::Value*>& inputs, const ir::Value* output) {
  CHECK_IF_NULL(output);

  for (size_t i = 0; i < input_ids.size(); ++i) {
    if (input_ids[i] >= inputs.size()) {
      RT_GLOG(EXCEPTION) << "Input index out of range: " << input_ids[i];
    }
    auto tensor = inputs[input_ids[i]]->ToTensor();
    CHECK_IF_NULL(tensor);
    UpdateAddress(tensor.get(), &(variant_pack.inTensors[i]));
  }

  std::vector<ir::TensorPtr> output_tensors;
  if (output->IsTuple()) {
    output_tensors = output->ToTuple()->ToTensorList();
  } else {
    auto tensor = output->ToTensor();
    CHECK_IF_NULL(tensor);
    output_tensors = {tensor};
  }

  for (size_t i = 0; i < output_ids.size(); ++i) {
    if (output_ids[i] >= output_tensors.size()) {
      RT_GLOG(EXCEPTION) << "Output index out of range: " << output_ids[i];
    }
    UpdateAddress(output_tensors[output_ids[i]].get(), &(variant_pack.outTensors[i]));
  }
}

AtbContextManager& AtbContextManager::GetInstance() {
  static AtbContextManager instance;
  return instance;
}

atb::Context* AtbContextManager::GetContext(const aclrtStream& stream) {
  auto& atb_context = context_map_[stream];
  if (atb_context == nullptr) {
    auto create_status = atb::CreateContext(&atb_context);
    if (create_status != 0) {
      RT_GLOG(EXCEPTION) << "Create atb context failed.";
    }
    auto set_status = atb_context->SetExecuteStream(stream);
    if (set_status != 0) {
      RT_GLOG(EXCEPTION) << "Set atb context stream failed.";
    }
  }
  return atb_context;
}

AtbContextManager::~AtbContextManager() {
  for (auto& item : context_map_) {
    if (item.second != nullptr) {
      auto destroy_status = atb::DestroyContext(item.second);
      if (destroy_status != 0) {
        std::cerr << "Destroy atb context failed: " << destroy_status << std::endl;
      }
    }
  }
}

atb::Context* GetAtbContext(const aclrtStream& stream) {
  return AtbContextManager::GetInstance().GetContext(stream);
}

} // namespace ops
} // namespace fxrt
