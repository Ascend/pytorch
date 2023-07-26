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
#include "torch_npu/csrc/framework/graph/util/GraphModeGuard.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/ForceJitCompileList.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"

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

static std::unordered_map<aclDataType, at::ScalarType>
    ACL_SCALAR_TYPE_TO_AT_TYPE_MAP = {
        {ACL_UINT8, at::ScalarType::Byte},
        {ACL_INT8, at::ScalarType::Char},
        {ACL_INT16, at::ScalarType::Short},
        {ACL_INT32, at::ScalarType::Int},
        {ACL_FLOAT16, at::ScalarType::Half},
        {ACL_FLOAT, at::ScalarType::Float},
        {ACL_BOOL, at::ScalarType::Bool},
        {ACL_INT64, at::ScalarType::Long},
        {ACL_DOUBLE, at::ScalarType::Double},
};

static std::map<const string, const aclDataType>
    STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP = {{"uint16", ACL_UINT16},
                                          {"uint8", ACL_UINT8},
                                          {"uint64", ACL_UINT64},
                                          {"string", ACL_STRING}};

aclError AclrtMemcpyAsyncParamCheck(void *dst, size_t destMax, const void *src,
                                    size_t count, aclrtMemcpyKind kind,
                                    aclrtStream stream) {
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    if (dst == nullptr || src == nullptr) {
      AT_ERROR("Dst ptr or Src ptr of aclrtMemcpyAsync is nullptr!",
               "Current run mode is graph mode, "
               "try to use torch.npu.disable_graph_mode() to fix this error.");
    }
  }

  auto ret = aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
  return ret;
}

aclError AclrtMemcpyParamCheck(void *dst, size_t destMax, const void *src,
                               size_t count, aclrtMemcpyKind kind) {
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    if (dst == nullptr || src == nullptr) {
      AT_ERROR("Dst ptr or Src ptr of aclrtMemcpy is nullptr!",
               "Current run mode is graph mode, "
               "try to use torch.npu.disable_graph_mode() to fix this error.");
    }
  }

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

at::ScalarType CalcuOpUtil::ConvertToATDataType(const aclDataType &acl_type) {
  auto iter = ACL_SCALAR_TYPE_TO_AT_TYPE_MAP.find(acl_type);
  if (iter == ACL_SCALAR_TYPE_TO_AT_TYPE_MAP.end()) {
    NPU_LOGE("Unsupport data type: %d.", static_cast<int32_t>(acl_type));
    return at::ScalarType::Undefined;
  }
  return iter->second;
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
  GraphModeGuard mode_guard(c10_npu::ModeKind::SINGLE_OP_MODE);
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
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    GraphExecutor::GetInstance().ConstructAndExecuteGraph();
  }

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
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    GraphExecutor::GetInstance().ConstructAndExecuteGraph();
  }

  void *dst_ptr = static_cast<void *>(
      static_cast<uint8_t *>(dst.first->data()) + dst.second);
  return AclrtMemcpyParamCheck(dst_ptr, dstMax, src, count, kind);
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(
    void *dst, size_t dstMax,
    const StorageAndOffsetMemSizePair &src,
    size_t count, aclrtMemcpyKind kind) {
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    GraphExecutor::GetInstance().ConstructAndExecuteGraph();
  }

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
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    GraphExecutor::GetInstance().ConstructAndExecuteGraph();
  }

  aclError ret = c10_npu::queue::LaunchAsyncCopyTask(
      dst.data_ptr(), dstMax, src.data_ptr(), count, kind);
  return ret;
}

aclError CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(
    const c10::StorageImpl &dst, size_t dstMax, void *src, size_t count,
    aclrtMemcpyKind kind) {
  if (c10_npu::NpuRunMode::IsGraphMode()) {
    GraphExecutor::GetInstance().ConstructAndExecuteGraph();
  }

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
  for (int i = 0; i < outputs.size(); i++) {
    if (!outputs[i].defined())
      continue;

    assert_no_internal_overlap(outputs[i]);

    for (int j = 0; j < inputs.size(); j++) {
      assert_no_partial_overlap(outputs[i], inputs[j]);
    }
  }
}

string CalcuOpUtil::GetReductionStr(int64_t reduction) {
  string reductionStr;
  if (reduction == at::Reduction::None) {
    reductionStr = "none";
  } else if (reduction == at::Reduction::Mean) {
    reductionStr = "mean";
  } else {
    reductionStr = "sum";
  }

  return reductionStr;
}

int64_t CalcuOpUtil::MakeWrapDim(int64_t dim, int64_t dim_post_expr) {
  if (dim_post_expr <= 0) {
    dim_post_expr = 1; // this will make range [-1, 0]
  }
  if (dim < 0) {
    dim += dim_post_expr;
  }
  return dim;
}

bool CalcuOpUtil::IsTransposeLastTwoDims(const at::Tensor &tensor) {
  if (tensor.dim() < 2 || tensor.dim() > 3) {
    return false;
  }
  int64_t numel = 1;
  auto storageSize = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)
                         ->get_npu_desc()
                         .storage_sizes_;

  for (int i = 0; i < storageSize.size(); i++) {
    numel *= storageSize[i];
  }

  int64_t dim1 = tensor.dim() - 1;
  int64_t dim2 = tensor.dim() - 2;

  auto tensor_desc =
      torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc();
  if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2) &&
      tensor.size(dim1) == tensor_desc.base_sizes_[dim2] &&
      tensor.size(dim2) == tensor_desc.base_sizes_[dim1] &&
      tensor.numel() == numel &&
      tensor_desc.base_sizes_.size() == tensor.dim()) {
    return true;
  } else {
    return false;
  }
}

bool CalcuOpUtil::IsMmTranspose(const at::Tensor &tensor) {
  if (tensor.dim() < 2 || tensor.dim() > 3) {
    return false;
  }
  int64_t dim1 = tensor.dim() - 1;
  int64_t dim2 = tensor.dim() - 2;

  if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2)) {
    return true;
  } else {
    return false;
  }
}

bool CalcuOpUtil::IsNdToNzOnTheFly(const at::Tensor &self, const at::Tensor &mat2) {
  const static int64_t kInnerAxisMinLimit = 128;
  const static int64_t kInnerAxisMaxLimit = 65535;
  if (self.dim() < 2 || mat2.dim() < 2) {
    return false;
  }
  // get inner axis of input after transpose.
  int64_t self_inner_axis = self.size(self.dim() - 1);
  int64_t self_outer_axis = self.size(self.dim() - 2);
  int64_t mat2_inner_axis = mat2.size(mat2.dim() - 1);
  int64_t mat2_outer_axis = mat2.size(mat2.dim() - 2);
  if (IsTransposeLastTwoDims(self)) {
    self_inner_axis = self.size(self.dim() - 2);
    self_outer_axis = self.size(self.dim() - 1);
  }
  if (IsTransposeLastTwoDims(mat2)) {
    mat2_inner_axis = mat2.size(mat2.dim() - 2);
    mat2_outer_axis = mat2.size(mat2.dim() - 1);
  }
  if (self_inner_axis * self_outer_axis <= kInnerAxisMaxLimit &&
      mat2_inner_axis * mat2_outer_axis <= kInnerAxisMaxLimit) {
    // too small tensor size
    return true;
  }
  // self inner_axis and mat2_inner_axis both in [128, 65535] or in (0, 128) and is multi of 16
  return ((self_inner_axis >= kInnerAxisMinLimit && self_inner_axis <= kInnerAxisMaxLimit) ||
          (self_inner_axis < kInnerAxisMinLimit && !(self_inner_axis & 0xF))) &&
         ((mat2_inner_axis >= kInnerAxisMinLimit && mat2_inner_axis <= kInnerAxisMaxLimit) ||
          (mat2_inner_axis < kInnerAxisMinLimit && !(mat2_inner_axis & 0xF)));
}

bool CalcuOpUtil::IsTransposeInnerAxis(const at::Tensor &self) {
  const static int64_t kInnerAxisMinBytes = 256;
  const static int64_t kInnerAxisMaxLimit = 65535;
  if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910B1 || self.dim() < 2 ||
      (self.scalar_type() != at::ScalarType::Half && self.scalar_type() != at::ScalarType::Float)) {
    return false;
  }
  int64_t data_type = elementSize(self.scalar_type());
  int64_t self_inner_axis = self.size(self.dim() - 1);
  int64_t self_outer_axis = self.size(self.dim() - 2);
  if (IsMmTranspose(self)) {
    self_inner_axis = self.size(self.dim() - 2);
    self_outer_axis = self.size(self.dim() - 1);
  }
  if (self_inner_axis == 1 && self_outer_axis > kInnerAxisMaxLimit) {
    return true;
  }
  if (self_inner_axis * self_outer_axis <= kInnerAxisMaxLimit) {
    // too small tensor size
    return false;
  }
  return ((self_inner_axis > kInnerAxisMaxLimit) ||
          (self_inner_axis * data_type < kInnerAxisMinBytes && bool((self_inner_axis * data_type) & 0x1F))) &&
         ((self_outer_axis * data_type >= kInnerAxisMinBytes && self_outer_axis <= kInnerAxisMaxLimit) ||
          (self_outer_axis * data_type < kInnerAxisMinBytes && !((self_outer_axis * data_type) & 0x1F)));
}

bool CalcuOpUtil::IsTransposeBothInnerAxis(const at::Tensor &self, const at::Tensor &mat2) {
  const static int64_t kInnerAxisMaxLimit = 65535;
  int64_t self_inner_axis = self.size(self.dim() - 1);
  int64_t self_outer_axis = self.size(self.dim() - 2);
  int64_t mat2_inner_axis = mat2.size(mat2.dim() - 1);
  int64_t mat2_outer_axis = mat2.size(mat2.dim() - 2);
  if (IsTransposeLastTwoDims(self)) {
    self_inner_axis = self.size(self.dim() - 2);
    self_outer_axis = self.size(self.dim() - 1);
  }
  if (IsTransposeLastTwoDims(mat2)) {
    mat2_inner_axis = mat2.size(mat2.dim() - 2);
    mat2_outer_axis = mat2.size(mat2.dim() - 1);
  }
  return self_inner_axis > kInnerAxisMaxLimit && self_outer_axis <= kInnerAxisMaxLimit &&
         mat2_inner_axis > kInnerAxisMaxLimit && mat2_outer_axis <= kInnerAxisMaxLimit;
}

void CalcuOpUtil::InsertInputPad(at::Tensor &self, at::Tensor &mat2) {
  bool is_self_trans = IsTransposeLastTwoDims(self);
  bool is_mat2_trans = IsTransposeLastTwoDims(mat2);
  int64_t m_dim = self.size(-2);
  int64_t n_dim = mat2.size(-1);
  int64_t k_dim = self.size(-1);
  int64_t data_size = elementSize(self.scalar_type());
  // k_dim less than is skipped
  const int64_t min_k_dim = 1024;
  // when k_dim exceeds 4096, pad + aligned matmul costs more than single unaligned matmul
  const int64_t max_k_dim = 4096;
  // 512B aligned shape is soc friendly
  const int64_t kPackage512 = 512;
  // one block takes 32 bytes
  const int64_t kBlockBytes = 32;
  bool valid_scenario = (m_dim * data_size) % kPackage512 == 0 && (n_dim * data_size) % kPackage512 == 0;
  valid_scenario &= (k_dim * data_size) % kBlockBytes != 0 && IsHalfFloatDtype(self);
  valid_scenario &= m_dim > k_dim && n_dim > k_dim && k_dim > min_k_dim && k_dim < max_k_dim;
  valid_scenario &= c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
  if (valid_scenario) {
    int64_t pad_num = Ceil(k_dim, CeilDiv(kPackage512, data_size)) - k_dim;
    // pad: left, right, top, bottom
    vector<int64_t> self_pad = {0, 0, 0, 0};
    vector<int64_t> mat2_pad = {0, 0, 0, 0};
    self_pad[2 * is_self_trans + 1] = pad_num;
    mat2_pad[2 * (1 - is_mat2_trans) + 1] = pad_num;
    self = is_self_trans ? self.transpose(-1, -2) : self;
    mat2 = is_mat2_trans ? mat2.transpose(-1, -2) : mat2;
    self = NPUNativeFunctions::constant_pad_nd(self, self_pad, 0);
    mat2 = NPUNativeFunctions::constant_pad_nd(mat2, mat2_pad, 0);
    self = is_self_trans ? self.transpose(-1, -2) : self;
    mat2 = is_mat2_trans ? mat2.transpose(-1, -2) : mat2;
  }
}

int64_t CalcuOpUtil::Ceil(int64_t x, int64_t y) {
  TORCH_CHECK(y != 0 , "Error, zero division.");
  return ((x + y - 1) / y) * y;
}

int64_t CalcuOpUtil::CeilDiv(int64_t x, int64_t y) {
  TORCH_CHECK(y != 0 , "Error, zero division.");
  return (x + y - 1) / y;
}

bool CalcuOpUtil::IsHalfFloatDtype(const at::Tensor &tensor) {
  return tensor.scalar_type() == at::ScalarType::Half || tensor.scalar_type() == at::ScalarType::BFloat16;
}

bool CalcuOpUtil::IsScalarWrappedToTensor(const at::Tensor &tensor) {
  return tensor.unsafeGetTensorImpl()->is_wrapped_number() &&
         (!at_npu::key::isDeviceTensor(tensor));
}

bool CalcuOpUtil::IsScalarOne(const c10::Scalar &scalar) {
  if (scalar.isIntegral(false)) {
    return scalar.toInt() == 1;
  } else if (scalar.isFloatingPoint()) {
    return fabs(scalar.toFloat() - 1.0) < EPSILON;
  } else {
    return false;
  }
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

c10::SmallVector<at::Tensor, N> CalcuOpUtil::ConvertTensorListToSmallVector(
    at::TensorList tensors) {
  c10::SmallVector<at::Tensor, N> tensorVec;
  for (int i = 0; i < tensors.size(); i++) {
    tensorVec.emplace_back(tensors[i]);
  }

  return tensorVec;
}

c10::SmallVector<int64_t, N> CalcuOpUtil::ConvertIntArrayRefToSmallVector(
    c10::IntArrayRef intArray) {
  c10::SmallVector<int64_t, N> intVec;
  for (int i = 0; i < intArray.size(); i++) {
    intVec.emplace_back(intArray[i]);
  }

  return intVec;
}

c10::SmallVector<int64_t, N> CalcuOpUtil::GetDimlistForTensor(
    const at::Tensor &self) {
  c10::SmallVector<int64_t, N> dimList = {};
  for (int64_t i = 0; i < self.dim(); i++) {
    dimList.emplace_back(i);
  }
  return dimList;
}

int64_t CalcuOpUtil::CompletePad(int64_t s_size, int64_t p_size, int64_t k_size,
                                 int64_t stride) {
  int64_t needpads = 0;
  int64_t sizeP = s_size + p_size * 2;
  int64_t leftLen = sizeP - k_size;
  if (stride == 0) {
    AT_ERROR("CompletePad stride is zero!");
  }
  auto reminder = leftLen % stride;
  if (reminder != 0) {
    needpads = stride - reminder;
  }
  return needpads;
}

c10::SmallVector<int64_t, 3> CalcuOpUtil::ComputeOutputSize(
    c10::IntArrayRef input_size, // Full input tensor size.
    c10::optional<c10::IntArrayRef> output_size,
    c10::optional<c10::ArrayRef<double>> scale_factors) {
  int spatial_dimensions = input_size.size() - 2;
  if (output_size) {
    TORCH_CHECK(!scale_factors,
                "Must specify exactly one of output_size and scale_factors");
    TORCH_CHECK(output_size->size() == spatial_dimensions);
    return {output_size->data(), output_size->data() + output_size->size()};
  }
  if (scale_factors) {
    TORCH_CHECK(!output_size,
                "Must specify exactly one of output_size and scale_factors");
    TORCH_CHECK(scale_factors->size() == spatial_dimensions);
    c10::SmallVector<int64_t, 3> ret;
    for (int i = 0; i < spatial_dimensions; ++i) {
      ret.push_back(static_cast<double>(input_size[i + 2]) *
                    scale_factors.value()[i]);
    }
    return ret;
  }
  TORCH_CHECK(false,
              "Must specify exactly one of output_size and scale_factors");
}

c10::optional<double> CalcuOpUtil::GetScaleValue(
    c10::optional<c10::ArrayRef<double>> scales, int idx) {
  if (!scales) {
    return c10::nullopt;
  }
  return scales->at(idx);
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
      storage_impl, storage_impl, dtype);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(c10::MemoryFormat::Contiguous);
  return tensor;
}

} // namespace native
} // namespace at_npu
