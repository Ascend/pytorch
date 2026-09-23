#include <torch/extension.h>
#include <vector>
#include <utility>
#ifdef ENABLE_TORCH_NPU
#include "acl/acl.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#endif
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "ir/common/intrusive_ptr.h"
#include "ir/tensor/tensor.h"
#include "ir/tensor/format.h"
#include "ops/utils/data_convert.h"
#include "ops/utils/utils.h"
#include "common/logger.h"

namespace ir = fxrt::ir;

namespace fxrt::ops {
using ir::MemoryFormat;

static const std::map<at::ScalarType, ir::DataType> kAtScalarTypeToDataTypeMap = {
    {at::kHalf, ir::DataType::Type::Float16},
    {at::kBFloat16, ir::DataType::Type::BFloat16},
    {at::kFloat, ir::DataType::Type::Float32},
    {at::kDouble, ir::DataType::Type::Float64},
    {at::kComplexFloat, ir::DataType::Type::Complex64},
    {at::kChar, ir::DataType::Type::Int8},
    {at::kShort, ir::DataType::Type::Int16},
    {at::kInt, ir::DataType::Type::Int32},
    {at::kLong, ir::DataType::Type::Int64},
    {at::kByte, ir::DataType::Type::UInt8},
    {at::kQInt8, ir::DataType::Type::QInt8},
    {at::kQUInt4x2, ir::DataType::Type::QUInt4x2},
    {at::kBool, ir::DataType::Type::Bool},
};

static const std::map<ir::DataType, at::ScalarType> kDataTypeToAtScalarTypeMap = {
    {ir::DataType::Type::Float16, at::kHalf},
    {ir::DataType::Type::BFloat16, at::kBFloat16},
    {ir::DataType::Type::Float32, at::kFloat},
    {ir::DataType::Type::Float64, at::kDouble},
    {ir::DataType::Type::Complex64, at::kComplexFloat},
    {ir::DataType::Type::Int8, at::kChar},
    {ir::DataType::Type::Int16, at::kShort},
    {ir::DataType::Type::Int32, at::kInt},
    {ir::DataType::Type::Int64, at::kLong},
    {ir::DataType::Type::UInt8, at::kByte},
    {ir::DataType::Type::QInt8, at::kQInt8},
    {ir::DataType::Type::QUInt4x2, at::kQUInt4x2},
    {ir::DataType::Type::Bool, at::kBool},
};

#ifdef ENABLE_TORCH_NPU
inline aclFormat ConvertMemoryFormatToAclFormat(ir::MemoryFormat format) {
  static const std::unordered_map<ir::MemoryFormat, aclFormat> kMemoryFormatToAclFormatMap = {
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
#endif

ir::DataType FromTorchDType(const at::ScalarType& type) {
  auto iter = kAtScalarTypeToDataTypeMap.find(type);
  if (iter == kAtScalarTypeToDataTypeMap.end()) {
    RT_GLOG(EXCEPTION) << "Unsupported at::ScalarType" << type << "for conversion to ir::DataType";
    return ir::DataType::Unknown;
  }

  return iter->second;
}

at::ScalarType ToTorchDType(ir::DataType type) {
  auto iter = kDataTypeToAtScalarTypeMap.find(type);
  if (iter == kDataTypeToAtScalarTypeMap.end()) {
    RT_GLOG(EXCEPTION) << "Unsupported ir::DataType " << type << " for conversion to at::ScalarType";
    return at::kFloat;
  }

  return iter->second;
}

// Device conversion utilities
hardware::Device FromTorchDevice(const at::Device& device) {
  hardware::DeviceType deviceType = hardware::DeviceType::CPU;
  switch (device.type()) {
    case at::DeviceType::CPU:
      deviceType = hardware::DeviceType::CPU;
      break;
    case at::DeviceType::PrivateUse1:
      deviceType = hardware::DeviceType::NPU;
      break;
    default:
      RT_GLOG(EXCEPTION) << "Unsupported torch::Device " << device.str() << " for conversion to hardware::Device";
  }
  return hardware::Device(deviceType, device.index());
}

at::Device ToTorchDevice(const hardware::Device device) {
  at::DeviceType deviceType = at::kCPU;
  switch (device.type) {
    case hardware::DeviceType::CPU:
      deviceType = at::kCPU;
      break;
#ifdef ENABLE_TORCH_NPU
    case hardware::DeviceType::NPU:
      deviceType = at::kPrivateUse1;
      break;
#endif
    default:
      RT_GLOG(EXCEPTION) << "Unsupported hardware::DeviceType " << hardware::GetDeviceNameByType(device.type)
                         << " for conversion to torch::Device";
  }
  return at::Device(deviceType, device.index);
}

// Create a new fxrt Tensor with a weak ref to torch Tensor data
ir::TensorPtr FromTorchTensor(const at::Tensor& tensor, bool isFake) {
  ir::DataType type = FromTorchDType(tensor.scalar_type());
  std::vector<int64_t> shape;
  shape.reserve(tensor.dim());
  for (auto& dim : tensor.sym_sizes()) {
    if (dim.is_symbolic()) {
      (void)shape.emplace_back(-1);
    } else if (dim.maybe_as_int().has_value()) {
      (void)shape.emplace_back(dim.maybe_as_int().value());
    } else {
      RT_GLOG(EXCEPTION) << "Dynamic shape with non-int dimension is not supported";
    }
  }

  auto device = FromTorchDevice(tensor.device());
  if (isFake) {
    return ir::MakeIntrusive<ir::Tensor>(shape, type, device);
  } else {
    return ir::MakeIntrusive<ir::Tensor>(tensor.data_ptr(), shape, type, device);
  }
}

// Create a new torch Tensor by moving ownership of data from fxrt Tensor
at::Tensor ToTorchTensor(const ir::TensorPtr& tensor) {
  CHECK_IF_NULL(tensor);
  if (!tensor->IsContiguous() && !IsTensorBaseFormat(tensor)) {
    RT_GLOG(EXCEPTION)
        << "Custom call input does not support non-contiguous tensor with non-base memory format, format: "
        << ir::FormatEnumToStr(tensor->Format()) << ", shape: " << tensor->Shape() << ", strides: " << tensor->Strides()
        << ", storageOffset: " << tensor->StorageOffset();
  }
  auto storage = tensor->GetStorage();
  void* dataPtr = storage->Data();
  void* blob_data_ptr = nullptr;
  // Allow nullptr for 0-element tensors; PyTorch treats this as valid.
  if (tensor->Numel() > 0) {
    CHECK_IF_NULL(dataPtr);
    blob_data_ptr = tensor->DataPtr();
  }

  auto atDevice = ToTorchDevice(tensor->GetDevice());
  auto options = at::TensorOptions().dtype(ToTorchDType(tensor->Dtype())).device(atDevice);

  switch (atDevice.type()) {
    case at::DeviceType::CPU: {
      if (tensor->Strides().empty()) {
        return at::from_blob(const_cast<void*>(blob_data_ptr), tensor->Shape(), options);
      }
      return at::from_blob(const_cast<void*>(blob_data_ptr), tensor->Shape(), tensor->Strides(), nullptr, options);
    }
#ifdef ENABLE_TORCH_NPU
    case at::DeviceType::PrivateUse1: {
      at::Tensor out;
      if (tensor->Strides().empty()) {
        out = at_npu::native::from_blob(const_cast<void*>(blob_data_ptr), tensor->Shape(), options);
      } else {
        out = at_npu::native::from_blob(
            const_cast<void*>(blob_data_ptr), tensor->Shape(), tensor->Strides(), 0, nullptr, options);
      }
      auto& desc = static_cast<torch_npu::NPUStorageImpl*>(out.storage().unsafeGetStorageImpl())->npu_desc_;
      desc.npu_format_ = ConvertMemoryFormatToAclFormat(tensor->Format());
      return out;
    }
#endif
    default:
      RT_GLOG(EXCEPTION) << "Unsupported DeviceType " << atDevice.str();
  }
  return at::Tensor{};
}

void* GetFirstTensorStorageData(const ir::Value* value) {
  if (value == nullptr) {
    return nullptr;
  }
  if (value->IsTensor()) {
    return value->ToTensor()->GetStorage()->Data();
  } else if (value->IsTuple() && value->ToTuple()->Size() > 0) {
    return GetFirstTensorStorageData((*value->ToTuple())[0].get());
  }
  return nullptr;
}

void CheckOutputInputRef(
    const std::vector<const ir::Value*>& input,
    const ir::Value* output,
    const std::string& opName) {
  if (input.size() > 0 && input[0]->IsTensor() && output != nullptr) {
    const auto* inputStorageData = input[0]->ToTensor()->GetStorage()->Data();
    void* outputStorageData = GetFirstTensorStorageData(output);
    if (outputStorageData == inputStorageData) {
      RT_GLOG(EXCEPTION) << "Custom/Python Call: Operator " << opName << " does not support reference. "
                         << "Input[0] Storage Data: " << inputStorageData << ", "
                         << "Output Storage Data: " << outputStorageData;
    }
  }
}

bool IsTorchTensorStandardLayout(const at::Tensor& atTensor) {
  if (!atTensor.is_contiguous() || atTensor.storage_offset() != 0) {
    return false;
  }
  switch (atTensor.device().type()) {
    case at::DeviceType::CPU:
      return true;
#ifdef ENABLE_TORCH_NPU
    case at::DeviceType::PrivateUse1:
      return IsBaseFormat(static_cast<ir::MemoryFormat>(at_npu::native::get_npu_format(atTensor)));
#endif
    default:
      RT_GLOG(EXCEPTION) << "Unsupported DeviceType " << atTensor.device().str();
  }
  return false;
}

void UpdateTensorFromTorch(const ir::TensorPtr& tensor, const at::Tensor& atTensor) {
#ifdef ENABLE_TORCH_NPU
  auto device = tensor->GetDevice();
  if (device.type == hardware::DeviceType::NPU) {
    auto npuFormat = at_npu::native::get_npu_format(atTensor);
    tensor->SetFormat(static_cast<ir::MemoryFormat>(npuFormat));
    tensor->SetStrides(atTensor.strides().vec());
    tensor->SetStorageOffset(atTensor.storage_offset());
    tensor->SetStorageShape(at_npu::native::get_npu_storage_sizes(atTensor));
    RT_VLOG(VL_OPS) << "Update tensor from torch, format=" << ir::FormatEnumToStr(tensor->Format())
                    << ", strides=" << tensor->Strides() << ", storageOffset=" << tensor->StorageOffset()
                    << ", storageShape=" << tensor->StorageShape() << ", isView=" << atTensor.is_view()
                    << " at.tensor.shape: " << atTensor.sizes();
  }
#else
  (void)tensor;
  (void)atTensor;
#endif
}

} // namespace fxrt::ops
