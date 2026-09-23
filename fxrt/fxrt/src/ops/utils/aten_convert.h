#ifndef __OPS_UTILS_ATEN_CONVERT_H__
#define __OPS_UTILS_ATEN_CONVERT_H__

#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <map>
#include <string>
#include <cstring>

#include "common/logger.h"
#include "ir/common/dtype.h"
#include "ir/value/value.h"
#include "ir/tensor/format.h"

#ifdef ENABLE_TORCH_NPU
#include "acl/acl.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#endif

namespace fxrt {
namespace ops {
static const std::map<ir::DataType, at::ScalarType> kDataTypeToAtScalarTypeMap = {
    {ir::DataType::Type::Float16, at::kHalf},
    {ir::DataType::Type::BFloat16, at::kBFloat16},
    {ir::DataType::Type::Float32, at::kFloat},
    {ir::DataType::Type::Float64, at::kDouble},
    {ir::DataType::Type::Int8, at::kChar},
    {ir::DataType::Type::Int16, at::kShort},
    {ir::DataType::Type::Int32, at::kInt},
    {ir::DataType::Type::Int64, at::kLong},
    {ir::DataType::Type::UInt8, at::kByte},
    {ir::DataType::Type::QInt8, at::kQInt8},
    {ir::DataType::Type::Bool, at::kBool},
    {ir::DataType::Type::QUInt4x2, at::kQUInt4x2},
};

inline at::ScalarType ToAtenDType(ir::DataType type) {
  auto iter = kDataTypeToAtScalarTypeMap.find(type);
  if (iter == kDataTypeToAtScalarTypeMap.end()) {
    RT_GLOG(EXCEPTION) << "Unsupported ir::DataType " << type << " for conversion to at::ScalarType";
    return at::kFloat;
  }

  return iter->second;
}

#ifdef ENABLE_TORCH_NPU
inline aclFormat ConvertMemoryFormatToAclFormat(ir::MemoryFormat format) {
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
#endif

inline at::Tensor ToAtenTensor(const ir::Value* value) {
  auto tensor = value->ToTensor();
  auto options = at::TensorOptions().dtype(ToAtenDType(tensor->Dtype()));
  auto device = tensor->GetDevice();
#ifdef ENABLE_TORCH_NPU
  if (device.type == fxrt::hardware::DeviceType::NPU) {
    options = options.device(at::Device(at::kPrivateUse1, device.index));
    auto out = at_npu::native::from_blob(const_cast<void*>(tensor->DataPtr()), tensor->Shape(), options);
    auto& desc = static_cast<torch_npu::NPUStorageImpl*>(out.storage().unsafeGetStorageImpl())->npu_desc_;
    desc.npu_format_ = ConvertMemoryFormatToAclFormat(tensor->Format());
    return out;
  }
#endif
  if (device.type == fxrt::hardware::DeviceType::CPU) {
    options = options.device(at::Device(at::kCPU, device.index));
    return at::from_blob(const_cast<void*>(tensor->DataPtr()), tensor->Shape(), options);
  }
  RT_GLOG(EXCEPTION) << "Unsupported DeviceType " << GetDeviceNameByType(device.type)
                     << " for conversion to at::Tensor";
  return at::empty({}, options);
}
} // namespace ops
} // namespace fxrt
#endif // __OPS_UTILS_ATEN_CONVERT_H__
