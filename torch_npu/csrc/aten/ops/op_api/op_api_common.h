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

#include <dlfcn.h>
#include <vector>
#include <functional>
#include <type_traits>
#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;

typedef aclTensor *(*_aclCreateTensor)(const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
                                       const int64_t *stride, int64_t offset, aclFormat format,
                                       const int64_t *storage_dims, uint64_t storage_dims_num,
                                       void *tensor_data);
typedef aclScalar *(*_aclCreateScalar)(void *value, aclDataType data_type);
typedef aclIntArray *(*_aclCreateIntArray)(const int64_t *value, uint64_t size);
typedef aclFloatArray *(*_aclCreateFloatArray)(const float *value, uint64_t size);
typedef aclBoolArray *(*_aclCreateBoolArray)(const bool *value, uint64_t size);
typedef aclTensorList *(*_aclCreateTensorList)(const aclTensor *const *value, uint64_t size);

typedef int (*_aclDestroyTensor)(const aclTensor *tensor);
typedef int (*_aclDestroyScalar)(const aclScalar *scalar);
typedef int (*_aclDestroyIntArray)(const aclIntArray *array);
typedef int (*_aclDestroyFloatArray)(const aclFloatArray *array);
typedef int (*_aclDestroyBoolArray)(const aclBoolArray *array);
typedef int (*_aclDestroyTensorList)(const aclTensorList *array);

#define GET_OP_API_FUNC(apiName)                                                             \
reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

inline const char *GetOpApiLibName(void) {
  return "libopapi.so";
}

inline const char *GetCustOpApiLibName(void) {
  return "libcust_opapi.so";
}

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName, const char *apiName) {
  auto funcAddr = dlsym(handler, apiName);
  if (funcAddr == nullptr) {
    ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
  }
  return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName) {
  auto handler = dlopen(libName, RTLD_LAZY);
  if (handler == nullptr) {
      ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
  }
  return handler;
}

inline void *GetOpApiFuncAddr(const char *apiName) {
  static auto custOpApiHandler = GetOpApiLibHandler(GetCustOpApiLibName());
  if (custOpApiHandler != nullptr) {
    auto funcAddr = GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
    if (funcAddr != nullptr) {
      return funcAddr;
    }
  }

  static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
  if (opApiHandler == nullptr) {
    return nullptr;
  }
  return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

inline aclTensor *ConvertType(const at::Tensor &at_tensor) {
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  if (!at_tensor.defined()) {
    return nullptr;
  }
  TORCH_CHECK(at_npu::key::isDeviceTensor(at_tensor), "only npu tensor is supported");
  at::ScalarType scalar_data_type = at_tensor.scalar_type();
  aclDataType acl_data_type = at_npu::native::CalcuOpUtil::ConvertToAclDataType(scalar_data_type);
  c10::SmallVector<int64_t, 5> storageDims;
  // if acl_data_type is ACL_STRING, storageDims is empty.
  if (acl_data_type != ACL_STRING) {
    storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
  }

  const auto dimNum = at_tensor.sizes().size();
  aclFormat format = ACL_FORMAT_ND;
  switch (dimNum) {
    case 3:format = ACL_FORMAT_NCL;
      break;
    case 4:format = ACL_FORMAT_NCHW;
      break;
    case 5:format = ACL_FORMAT_NCDHW;
      break;
    default:format = ACL_FORMAT_ND;
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
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  if (aclCreateScalar == nullptr) {
    return nullptr;
  }

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
  static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
  if (aclCreateIntArray == nullptr) {
    return nullptr;
  }
  auto array = aclCreateIntArray(at_array.data(), at_array.size());
  return array;
}

template<std::size_t N>
inline aclBoolArray *ConvertType(const std::array<bool, N> &value) {
  static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
  if (aclCreateBoolArray == nullptr) {
    return nullptr;
  }

  auto array = aclCreateBoolArray(value.data(), value.size());
  return array;
}

inline aclBoolArray *ConvertType(const at::ArrayRef<bool> &value) {
  static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
  if (aclCreateBoolArray == nullptr) {
    return nullptr;
  }

  auto array = aclCreateBoolArray(value.data(), value.size());
  return array;
}

inline aclTensorList *ConvertType(const at::TensorList &at_tensor_list) {
  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  if (aclCreateTensorList == nullptr) {
    return nullptr;
  }

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

inline aclDataType ConvertType(const at::ScalarType scalarType) {
  return at_npu::native::CalcuOpUtil::ConvertToAclDataType(scalarType);
}

template<typename T>
T ConvertType(T value) {
  return value;
}

inline void Release(aclTensor *p) {
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  if (aclDestroyTensor == nullptr) {
    return;
  }
  aclDestroyTensor(p);
}

inline void Release(aclScalar *p) {
  static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
  if (aclDestroyScalar == nullptr) {
    return;
  }
  aclDestroyScalar(p);
}

inline void Release(aclIntArray *p) {
  static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
  if (aclDestroyIntArray == nullptr) {
    return;
  }

  aclDestroyIntArray(p);
}

inline void Release(aclBoolArray *p) {
  static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
  if (aclDestroyBoolArray == nullptr) {
    return;
  }

  aclDestroyBoolArray(p);
}

inline void Release(aclTensorList *p) {
  static const auto aclDestroyTensorList = GET_OP_API_FUNC(aclDestroyTensorList);
  if (aclDestroyTensorList == nullptr) {
    return;
  }

  aclDestroyTensorList(p);
}

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

template<typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr, std::index_sequence<I...>) {
  typedef int(*OpApiFunc)(typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  return func;
}

template<typename Tuple>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

#define DO_COMPATIBILITY(aclnn_api, originCallExpression)                                                    \
  do {                                                                                                       \
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");            \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                          \
    if (getWorkspaceSizeFuncAddr == nullptr || opApiFuncAddr == nullptr) {                                   \
      ASCEND_LOGW("%s or %sGetWorkspaceSize not in %s, or %s not found. Will call %s",                       \
                  #aclnn_api, #aclnn_api, GetOpApiLibName(), GetOpApiLibName(), #originCallExpression);      \
      return originCallExpression;                                                                           \
    }                                                                                                        \
} while(0)

typedef int(*InitHugeMemThreadLocal)(void*, bool);
typedef void(*UnInitHugeMemThreadLocal)(void*, bool);
typedef void(*ReleaseHugeMem)(void*, bool);

/**
 * 异步调用npu执行, 无返回值.
 */
#define EXEC_NPU_CMD(aclnn_api, ...)                                                                         \
  do {                                                                                                       \
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");            \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                          \
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                              \
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                          \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                   \
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,                             \
                #aclnn_api, " or ", #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ",   \
                GetOpApiLibName(), "not found.");                                                            \
    TORCH_CHECK(at_npu::native::env::CheckForbidInternalFormat(), #aclnn_api ": Internal format not support,"\
                " please set 'torch.npu.config.allow_internal_format=False'");                               \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                          \
    uint64_t workspace_size = 0;                                                                             \
    uint64_t *workspace_size_addr = &workspace_size;                                                         \
    aclOpExecutor *executor = nullptr;                                                                       \
    aclOpExecutor **executor_addr = &executor;                                                               \
    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);              \
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);      \
    if (initMemFunc) {                                                                                       \
      initMemFunc(nullptr, false);                                                                           \
    }                                                                                                        \
    auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                   \
    static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);       \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                    \
    TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());         \
    void *workspace_addr = nullptr;                                                                          \
    if (workspace_size != 0) {                                                                               \
      auto workspace_tensor = CalcuOpUtil::UnsafeEmptyWorkspace(workspace_size);                             \
      workspace_addr = workspace_tensor.storage().data();                                                    \
    }                                                                                                        \
    auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int {      \
      typedef int(*OpApiFunc)(void*, uint64_t, aclOpExecutor*, const aclrtStream);                           \
      OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                      \
      auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                        \
      TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());                \
      ReleaseConvertTypes(converted_params);                                                                 \
      ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                      \
      if (releaseMemFunc) {                                                                                  \
        releaseMemFunc(nullptr, false);                                                                      \
      }                                                                                                      \
      return api_ret;                                                                                        \
    };                                                                                                       \
    at_npu::native::OpCommand cmd;                                                                           \
    cmd.Name(#aclnn_api);                                                                                    \
    cmd.SetCustomHandler(acl_call);                                                                          \
    cmd.Run();                                                                                               \
    if (unInitMemFunc) {                                                                                     \
        unInitMemFunc(nullptr, false);                                                                       \
    }                                                                                                        \
  } while (false)

#define EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnn_api, ...)                                                         \
  do {                                                                                                       \
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");            \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                          \
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                              \
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                          \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                   \
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,                             \
                #aclnn_api, " or ", #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ",   \
                GetOpApiLibName(), "not found.");                                                            \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                          \
    uint64_t workspace_size = 0;                                                                             \
    uint64_t *workspace_size_addr = &workspace_size;                                                         \
    aclOpExecutor *executor = nullptr;                                                                       \
    aclOpExecutor **executor_addr = &executor;                                                               \
    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);              \
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);      \
    if (initMemFunc) {                                                                                       \
      initMemFunc(nullptr, false);                                                                           \
    }                                                                                                        \
    auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                   \
    static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);       \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                    \
    TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());         \
    void *workspace_addr = nullptr;                                                                          \
    if (workspace_size != 0) {                                                                               \
      int deviceIdx = 0;                                                                                     \
      NPU_CHECK_ERROR(aclrtGetDevice(&deviceIdx));                                                           \
      auto workspace_tensor =                                                                                \
          at::empty({static_cast<int64_t>(workspace_size)},                                                  \
                    at::TensorOptions().device(at_npu::key::NativeDeviceType, deviceIdx).dtype(at::kByte));  \
      workspace_addr = workspace_tensor.storage().data();                                                    \
    }                                                                                                        \
    auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int {      \
      typedef int(*OpApiFunc)(void*, uint64_t, aclOpExecutor*, const aclrtStream);                           \
      OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                      \
      auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                        \
      TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());                \
      ReleaseConvertTypes(converted_params);                                                                 \
      ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                      \
      if (releaseMemFunc) {                                                                                  \
        releaseMemFunc(nullptr, false);                                                                      \
      }                                                                                                      \
      return api_ret;                                                                                        \
    };                                                                                                       \
    at_npu::native::OpCommand cmd;                                                                           \
    cmd.Name(#aclnn_api);                                                                                    \
    cmd.SetCustomHandler(acl_call);                                                                          \
    cmd.Run();                                                                                               \
    if (unInitMemFunc) {                                                                                     \
        unInitMemFunc(nullptr, false);                                                                       \
    }                                                                                                        \
  } while (false)

template<typename Tuple>
class ConvertedParams {
public:
  ConvertedParams(Tuple &&convertedParams) : convertedParams_(std::move(convertedParams)) {};
  ConvertedParams(ConvertedParams &&other) : convertedParams_(std::move(other.convertedParams_)) {
    other.validParams_ = false;
  };
  ConvertedParams &operator=(ConvertedParams &&other) {
    if (this == &other) {
      return *this;
    }

    convertedParams_ = std::move(other.convertedParams_);
    validParams_ = true;
    other.validParams_ = false;
    return *this;
  }

  ConvertedParams() = delete;
  ConvertedParams(const ConvertedParams &other) = delete;
  ConvertedParams &operator=(const ConvertedParams &other) = delete;

  ~ConvertedParams() {
    if (validParams_) {
      ReleaseConvertTypes(convertedParams_);
    }
  }

  const Tuple &GetConvertedParams() const {
    return convertedParams_;
  }

  template<size_t i>
  auto Get() {
    return std::get<i>(convertedParams_);
  }
private:
  Tuple convertedParams_;
  bool validParams_{true};
};

/**
 * 同步调用npu执行，返回把aten的tensor, scalar, array等转换后的参数,
 */
#define EXEC_NPU_CMD_SYNC(aclnn_api, ...)                                                                   \
  [](const char *apiName, const char *workspaceSizeApiName, auto&...args)->auto  {                          \
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(workspaceSizeApiName);                    \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(apiName);                                            \
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,                            \
                #aclnn_api, " and ", #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", \
                GetOpApiLibName(), "not found.");                                                           \
    TORCH_CHECK(at_npu::native::env::CheckForbidInternalFormat(),                                           \
                apiName, ": Internal format not support,"                                                   \
                         " please set 'torch.npu.config.allow_internal_format=False'");                     \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                         \
    uint64_t workspace_size = 0;                                                                            \
    uint64_t *workspace_size_addr = &workspace_size;                                                        \
    aclOpExecutor *executor = nullptr;                                                                      \
    aclOpExecutor **executor_addr = &executor;                                                              \
    auto converted_params = ConvertTypes(args..., workspace_size_addr, executor_addr);                      \
    static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);      \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                   \
    TORCH_CHECK(workspace_status == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());        \
    void *workspace_addr = nullptr;                                                                         \
    if (workspace_size != 0) {                                                                              \
      int deviceIdx = 0;                                                                                    \
      NPU_CHECK_ERROR(aclrtGetDevice(&deviceIdx));                                                          \
      auto workspace_tensor =                                                                               \
          at::empty({static_cast<int64_t>(workspace_size)},                                                 \
                    at::TensorOptions().device(at_npu::key::NativeDeviceType, deviceIdx).dtype(at::kByte)); \
      workspace_addr = workspace_tensor.storage().data();                                                   \
    }                                                                                                       \
    auto acl_call =                                                                                         \
        [converted_params, workspace_addr, workspace_size, acl_stream, executor, apiName]() -> int {        \
          typedef int(*OpApiFunc)(void *, uint64_t, aclOpExecutor *, const aclrtStream);                    \
          OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                 \
          auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                   \
          TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());           \
          return api_ret;                                                                                   \
        };                                                                                                  \
    at_npu::native::OpCommand cmd;                                                                          \
    cmd.Name(apiName);                                                                                      \
    cmd.SetCustomHandler(acl_call);                                                                         \
    cmd.Run();                                                                                              \
    cmd.Sync();                                                                                             \
    return ConvertedParams<decltype(converted_params)>(std::move(converted_params));                        \
  }(#aclnn_api, #aclnn_api "GetWorkspaceSize", __VA_ARGS__)

#endif //  TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_
