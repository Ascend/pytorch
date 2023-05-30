// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#ifndef TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_
#define TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_

#include <vector>
#include <functional>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <chrono>
#include <iostream>
#include <third_party/acl/inc/acl/acl_op_api.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"

inline aclTensor *ConvertType(const at::Tensor &at_tensor) {
  if (!at_tensor.defined()) {
    return nullptr;
  }
  TORCH_CHECK(!at_npu::native::CalcuOpUtil::IsScalarWrappedToTensor(at_tensor), "scalar wrapped tensor is unsupported");
  at::ScalarType scalar_data_type = at_tensor.scalar_type();
  aclDataType acl_data_type = at_npu::native::CalcuOpUtil::ConvertToAclDataType(scalar_data_type);
  c10::SmallVector<int64_t, 5> storageDims;
  // if acl_data_type is ACL_STRING, storageDims is empty.
  if (acl_data_type != ACL_STRING) {
    storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
  }

  aclFormat format = ACL_FORMAT_ND;
  if (at_tensor.sizes().size() == 3) {
    format = ACL_FORMAT_NCL;
  }

  if (at_tensor.sizes().size() == 4) {
    format = ACL_FORMAT_NCHW;
  }

  if (at_tensor.sizes().size() == 5) {
    format = ACL_FORMAT_NCDHW;
  }

  if (at_npu::native::CalcuOpUtil::IsScalarWrappedToTensor(at_tensor)) {
    c10::Scalar expScalar = at_npu::native::CalcuOpUtil::ConvertTensorToScalar(at_tensor);
    at::Tensor aclInput = at_npu::native::CalcuOpUtil::CopyScalarToDevice(expScalar, scalar_data_type);
    return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), acl_data_type,
                           aclInput.strides().data(), aclInput.storage_offset(), format, storageDims.data(),
                           storageDims.size(), aclInput.storage().data());
  }

  auto acl_tensor = aclCreateTensor(at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type,
                                    at_tensor.strides().data(), at_tensor.storage_offset(), format, storageDims.data(),
                                    storageDims.size(), at_tensor.storage().data());
  return acl_tensor;
}

inline aclScalar *ConvertType(const at::Scalar &at_scalar) {
  at::ScalarType scalar_data_type = at_scalar.type();
  aclDataType acl_data_type = at_npu::native::CalcuOpUtil::ConvertToAclDataType(scalar_data_type);
  aclScalar *acl_scalar = nullptr;
  switch (scalar_data_type) {
    case at::ScalarType::Double: {
      double value = at_scalar.toDouble();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::Long: {
      int64_t value = at_scalar.toLong();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::Bool: {
      bool value = at_scalar.toBool();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::ComplexDouble: {
      auto value = at_scalar.toComplexDouble();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    default:acl_scalar = nullptr;
      break;
  }

  return acl_scalar;
}

inline aclIntArray *ConvertType(const at::IntArrayRef &at_array) {
  auto array = aclCreateIntArray(at_array.data(), at_array.size());
  return array;
}

template<std::size_t N>
inline aclBoolArray *ConvertType(const std::array<bool, N>& value) {
  auto array = aclCreateBoolArray(value.data(), value.size());
  return array;
}

inline aclBoolArray *ConvertType(const at::ArrayRef<bool>& value) {
  auto array = aclCreateBoolArray(value.data(), value.size());
  return array;
}

inline aclTensorList *ConvertType(const at::TensorList &at_tensor_list) {
  std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
  for (size_t i = 0; i < at_tensor_list.size(); i++) {
    tensor_list[i] = ConvertType(at_tensor_list[i]);
  }
  auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
  return acl_tensor_list;
}

inline aclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor) {
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    return ConvertType(opt_tensor.value());
  }

  return nullptr;
}

inline aclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &opt_array) {
  if (opt_array.has_value()) {
    return ConvertType(opt_array.value());
  }

  return nullptr;
}

inline aclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar) {
  if (opt_scalar.has_value()) {
    return ConvertType(opt_scalar.value());
  }

  return nullptr;
}

inline aclDataType ConvertType(at::ScalarType scalarType) {
  return at_npu::native::CalcuOpUtil::ConvertToAclDataType(scalarType);
}

template<typename T>
T ConvertType(T value) {
  return value;
}

inline void Release(aclTensor *p) { aclDestroyTensor(p); }
inline void Release(aclScalar *p) { aclDestroyScalar(p); }
inline void Release(aclIntArray *p) { aclDestroyIntArray(p); }
inline void Release(aclBoolArray *p) { aclDestroyBoolArray(p); }
inline void Release(aclTensorList *p) { aclDestroyTensorList(p); }

template<typename T>
void Release(T value) {
  (void) value;
}

template<typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>) {
  (void) std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template<typename Tuple>
void ReleaseConvertTypes(Tuple &t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  CallRelease(t, std::make_index_sequence<size>{});
}

template<typename... Ts>
constexpr auto ConvertTypes(Ts &...args) {
  return std::make_tuple(ConvertType(args)...);
}

template<typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template<typename Function, typename Tuple>
auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

inline float duration(const std::chrono::steady_clock::time_point &end,
                      const std::chrono::steady_clock::time_point &begin) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1000.0;
}

/**
 *
 */
#define EXEC_NPU_CMD(aclnn_api, ...)                                                                         \
  do {                                                                                                       \
    TORCH_CHECK(at_npu::native::env::CheckForbidInternalFormat(), #aclnn_api ": Internal format not support,"\
                " please set 'torch.npu.config.allow_internal_format=False'");                               \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream();                                               \
    uint64_t workspace_size = 0;                                                                             \
    uint64_t *workspace_size_addr = &workspace_size;                                                         \
    aclOpExecutor *executor = nullptr;                                                                       \
    aclOpExecutor **executor_addr = &executor;                                                               \
    auto begin = std::chrono::steady_clock::now();                                                           \
    auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                   \
    auto end = std::chrono::steady_clock::now();                                                             \
    begin = std::chrono::steady_clock::now();                                                                \
    auto workspace_status = call(aclnn_api##GetWorkspaceSize, converted_params);                             \
    end = std::chrono::steady_clock::now();                                                                  \
    begin = std::chrono::steady_clock::now();                                                                \
    TORCH_CHECK(workspace_status == OK, #aclnn_api, " Get workspace size failed, errno:", workspace_status); \
    void *workspace_addr = nullptr;                                                                          \
    if (workspace_size != 0) {                                                                               \
      auto workspace_tensor =                                                                                \
          at::empty({static_cast<int64_t>(workspace_size)},                                                  \
                    at::TensorOptions().device(at_npu::key::NativeDeviceType, -1).dtype(at::kByte));         \
      workspace_addr = workspace_tensor.storage().data();                                                    \
    }                                                                                                        \
    auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int {      \
      auto begin_2 = std::chrono::steady_clock::now();                                                       \
      auto api_ret = aclnn_api(workspace_addr, workspace_size, executor, acl_stream);                        \
      TORCH_CHECK(api_ret == OK, #aclnn_api, " run failed, errno:", api_ret);                                \
      ReleaseConvertTypes(converted_params);                                                                 \
      auto end_2 = std::chrono::steady_clock::now();                                                         \
      return api_ret;                                                                                        \
    };                                                                                                       \
    end = std::chrono::steady_clock::now();                                                                  \
    begin = std::chrono::steady_clock::now();                                                                \
    at_npu::native::OpCommand cmd;                                                                           \
    cmd.Name(#aclnn_api);                                                                                    \
    end = std::chrono::steady_clock::now();                                                                  \
    begin = std::chrono::steady_clock::now();                                                                \
    cmd.SetCustomHandler(acl_call);                                                                          \
    end = std::chrono::steady_clock::now();                                                                  \
    begin = std::chrono::steady_clock::now();                                                                \
    cmd.Run();                                                                                               \
    end = std::chrono::steady_clock::now();                                                                  \
  } while (false)

#endif //  TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_
