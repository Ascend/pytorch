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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include <unordered_map>

#include <ATen/record_function.h>
#include <c10/util/Exception.h>

#include "torch_npu/csrc/aten/mirror/NPUMemoryOverlap.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/framework/contiguous/ReshapeOpt.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/ForceJitCompileList.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "op_plugin/OpInterface.h"

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"

namespace {
constexpr float EPSILON = 1e-6;

// check all at::ScalarType is not negative
#define ENUM_PAIR_FUNC(_1, n)                                                  \
  static_assert(static_cast<int64_t>(at::ScalarType::n) >= 0,                  \
                #n " is negative");
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(ENUM_PAIR_FUNC)
#undef ENUM_PAIR_FUNC

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)                            \
  _(at::ScalarType::Byte, ACL_UINT8)                                           \
  _(at::ScalarType::Char, ACL_INT8)                                            \
  _(at::ScalarType::Short, ACL_INT16)                                          \
  _(at::ScalarType::Int, ACL_INT32)                                            \
  _(at::ScalarType::Long, ACL_INT64)                                           \
  _(at::ScalarType::Half, ACL_FLOAT16)                                         \
  _(at::ScalarType::Float, ACL_FLOAT)                                          \
  _(at::ScalarType::Double, ACL_DOUBLE)                                        \
  _(at::ScalarType::ComplexHalf, ACL_DT_UNDEFINED)                             \
  _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)                               \
  _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)                             \
  _(at::ScalarType::Bool, ACL_BOOL)                                            \
  _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)                                   \
  _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)                                  \
  _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)                                  \
  _(at::ScalarType::BFloat16, ACL_BF16)                                        \
  _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)                                \
  _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)                                \
  _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)                               \
  _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable
    [static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
        AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// check at::ScalarType has been changed or not
#define ENUM_PAIR_FUNC(at_dtype, acl_dtype)                                    \
  static_assert(                                                               \
      kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at_dtype)] ==     \
          (acl_dtype),                                                         \
      #at_dtype " and " #acl_dtype " is not match any more, please check "       \
                "AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR and modify it");
AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(ENUM_PAIR_FUNC)
#undef DEFINE_ENUM

static std::map<const string, const aclDataType>
    STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP = {{"uint16", ACL_UINT16},
                                          {"uint8", ACL_UINT8},
                                          {"uint64", ACL_UINT64},
                                          {"string", ACL_STRING}};

aclError AclrtMemcpyAsyncParamCheck(void *dst, size_t destMax, const void *src,
                                    size_t count, aclrtMemcpyKind kind,
                                    aclrtStream stream) {
  auto ret = aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
  return ret;
}

aclError AclrtMemcpyParamCheck(void *dst, size_t destMax, const void *src,
                               size_t count, aclrtMemcpyKind kind) {
  auto ret = aclrtMemcpy(dst, destMax, src, count, kind);
  return ret;
}
} // namespace

namespace at_npu {
namespace native {
aclDataType CalcuOpUtil::ConvertToAclDataType(const at::ScalarType &data_type) {
  auto acl_dtype =
      kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
  TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED,
              std::string(c10::toString(data_type)) + " has not been supported")
  return acl_dtype;
}

aclDataType CalcuOpUtil::ConvertToAclDataType(
    const at::ScalarType &data_type,
    const string &realDataType) {
  auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
  TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED,
              std::string(c10::toString(data_type)) + " has not been supported")
  if (!realDataType.empty()) {
    return STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP[realDataType];
  }
  return acl_dtype;
}

c10::Scalar CalcuOpUtil::ConvertTensorToScalar(const at::Tensor &tensor) {
  c10::Scalar expScalar;
  const at::Tensor *aclInput = &tensor;
  if (aclInput->scalar_type() == at::ScalarType::Double) {
    double value = *(double *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Long) {
    int64_t value = *(int64_t *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Float) {
    float value = *(float *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Int) {
    int value = *(int *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Half) {
    c10::Half value = *(c10::Half *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::Bool) {
    int8_t value = *(int8_t *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::ComplexDouble) {
    c10::complex<double> value = *(c10::complex<double> *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::ComplexFloat) {
    c10::complex<float> value = *(c10::complex<float> *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else if (aclInput->scalar_type() == at::ScalarType::BFloat16) {
    c10::BFloat16 value = *(c10::BFloat16 *)aclInput->data_ptr();
    c10::Scalar scalar(value);
    expScalar = scalar;
  } else {
    NPU_LOGE("unsupport scalar type! ");
    NPU_CHECK_ERROR(ACL_ERROR_UNSUPPORTED_DATA_TYPE);
  }

  return expScalar;
}

at::Tensor CalcuOpUtil::CopyScalarToDevice(const c10::Scalar &cpu_scalar,
                                           at::ScalarType scalar_data_type) {
  return CalcuOpUtil::CopyTensorHostToDevice(
      scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

at::Tensor CalcuOpUtil::CopyTensorHostToDevice(const at::Tensor &cpu_tensor) {
  at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
  int deviceIndex = 0;
  NPU_CHECK_ERROR(aclrtGetDevice(&deviceIndex));
  return cpuPinMemTensor.to(
      c10::Device(at_npu::key::NativeDeviceType, deviceIndex),
      cpuPinMemTensor.scalar_type(), true, true);
}

NPUStatus CalcuOpUtil::AclrtMemcpyAsync(
    const std::pair<at::Tensor, int64_t> &dst,
    size_t dst_size,
    const std::pair<at::Tensor, int64_t> &src,
    size_t src_size, aclrtMemcpyKind kind) {
  void *dst_ptr = reinterpret_cast<uint8_t *>(dst.first.data_ptr()) +
                  dst.second * dst.first.itemsize();
  void *src_ptr = reinterpret_cast<uint8_t *>(src.first.data_ptr()) +
                  src.second * src.first.itemsize();
  NPU_CHECK_ERROR(c10_npu::queue::LaunchAsyncCopyTask(
      dst_ptr, dst_size, const_cast<void *>(src_ptr), src_size, kind));

  return "SUCCESS";
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(
    const StorageAndOffsetMemSizePair &dst,
    size_t dstMax,
    const StorageAndOffsetMemSizePair &src,
    size_t count, aclrtMemcpyKind kind) {
  void *dst_ptr = static_cast<void *>(
      static_cast<uint8_t *>(dst.first->data()) + dst.second);
  void *src_ptr = static_cast<void *>(
      static_cast<uint8_t *>(src.first->data()) + src.second);
  return AclrtMemcpyParamCheck(dst_ptr, dstMax, const_cast<void *>(src_ptr),
                               count, kind);
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(
    const StorageAndOffsetMemSizePair &dst,
    size_t dstMax, const void *src,
    size_t count, aclrtMemcpyKind kind) {
  void *dst_ptr = static_cast<void *>(
      static_cast<uint8_t *>(dst.first->data()) + dst.second);
  return AclrtMemcpyParamCheck(dst_ptr, dstMax, src, count, kind);
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(
    void *dst, size_t dstMax,
    const StorageAndOffsetMemSizePair &src,
    size_t count, aclrtMemcpyKind kind) {
  void *src_ptr = static_cast<void *>(
      static_cast<uint8_t *>(src.first->data()) + src.second);
  return AclrtMemcpyParamCheck(dst, dstMax, const_cast<void *>(src_ptr), count,
                               kind);
}

aclError CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(const at::Tensor &dst,
                                                        size_t dstMax,
                                                        const at::Tensor &src,
                                                        size_t count,
                                                        aclrtMemcpyKind kind) {
  aclError ret = c10_npu::queue::LaunchAsyncCopyTask(
      dst.data_ptr(), dstMax, src.data_ptr(), count, kind);
  return ret;
}

aclError CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(
    const c10::StorageImpl &dst, size_t dstMax, void *src, size_t count,
    aclrtMemcpyKind kind) {
  aclError ret =
      c10_npu::queue::LaunchAsyncCopyTask(dst.data(), dstMax, src, count, kind);
  return ret;
}

int64_t CalcuOpUtil::GetTensorNpuFormat(const at::Tensor &tensor) {
  TORCH_CHECK(tensor.device().type() == at_npu::key::NativeDeviceType,
              "Expected all tensors to be on the same device. "
              "Expected NPU tensor, please check whether the input tensor "
              "device is correct.");
  if (NpuUtils::check_match(&tensor) || NpuUtils::check_5d_5d_match(tensor)) {
    const torch_npu::NPUStorageDesc &tensor_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    return tensor_desc.npu_format_;
  } else {
    return InferFormat::GuessFormatWhenContiguous(tensor);
  }
}

void CalcuOpUtil::CheckMemoryOverLaps(
    c10::ArrayRef<at::Tensor> inputs,
    c10::ArrayRef<at::Tensor> outputs) {
  for (const auto i : c10::irange(outputs.size())) {
    if (!outputs[i].defined())
      continue;

    assert_no_internal_overlap(outputs[i]);

    for (const auto j : c10::irange(inputs.size())) {
      assert_no_partial_overlap(outputs[i], inputs[j]);
    }
  }
}

bool CalcuOpUtil::IsScalarWrappedToTensor(const at::Tensor &tensor) {
  return tensor.unsafeGetTensorImpl()->is_wrapped_number() &&
         (!at_npu::key::isDeviceTensor(tensor));
}

float CalcuOpUtil::GetScalarFloatValue(const c10::Scalar &scalar) {
  float value;
  if (scalar.isFloatingPoint()) {
    value = scalar.toFloat();
  } else {
    value = (float)scalar.toInt();
  }

  return value;
}

c10::SmallVector<int64_t, SHAPE_SIZE> CalcuOpUtil::ConvertIntArrayRefToSmallVector(
    c10::IntArrayRef intArray) {
  c10::SmallVector<int64_t, SHAPE_SIZE> intVec;
  for (const auto i : c10::irange(intArray.size())) {
    intVec.emplace_back(intArray[i]);
  }

  return intVec;
}

at::Tensor CalcuOpUtil::UnsafeEmptyWorkspace(uint64_t workspace_size) {
  ASCEND_LOGD("Alloc workspace %zu bytes unsafely.", workspace_size);
  c10::Allocator *allocator = c10_npu::NPUCachingAllocator::get();
  c10::intrusive_ptr<c10::StorageImpl> storage_impl =
      c10::make_intrusive<torch_npu::NPUStorageImpl>(
        c10::StorageImpl::use_byte_size_t(), workspace_size,
        allocator->allocate(workspace_size), allocator, true);
  static auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(at::kByte));
  auto tensor = at::detail::make_tensor<torch_npu::NPUTensorImpl>(
      storage_impl, dtype);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(c10::MemoryFormat::Contiguous);
  return tensor;
}

using aclCubeMathType = enum:int8_t {
  KEEP_DTYPE = 0,
  ALLOW_FP32_DOWN_PRECISION = 1,
  USE_FP16 = 2,
  USE_HF32 = 3,
};

static std::unordered_map<uint8_t, aclCubeMathType>
    ACL_CUBE_MATH_TYPE_MAP = {
        {0b00, KEEP_DTYPE},
        {0b01, USE_FP16},
        {0b10, USE_HF32},
        {0b11, ALLOW_FP32_DOWN_PRECISION}
    };

int8_t CalcuOpUtil::GetCubeMathType(bool allowHf32) {
  bool allowFp32ToFp16 = native::env::IsAllowFP32ToFP16();
  uint8_t CubeMathTypeCode = ((uint8_t)allowHf32 << 1) + (uint8_t)allowFp32ToFp16;
  auto iter = ACL_CUBE_MATH_TYPE_MAP.find(CubeMathTypeCode);
  if (iter == ACL_CUBE_MATH_TYPE_MAP.end()) {
    return ALLOW_FP32_DOWN_PRECISION;
  }
  return iter->second;
}

} // namespace native
} // namespace at_npu
