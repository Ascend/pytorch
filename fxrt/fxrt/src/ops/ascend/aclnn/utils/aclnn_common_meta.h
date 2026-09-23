#ifndef __OPS_ASCEND_ACLNN_UTILS_ACLNN_COMMON_META_H__
#define __OPS_ASCEND_ACLNN_UTILS_ACLNN_COMMON_META_H__

#include "common/visible.h"
#include "acl/acl_base.h"

namespace fxrt {
namespace ops {

// Base acl data structure
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;

// Base acl creators
using _aclCreateTensorFuncPtr = aclTensor* (*)(const int64_t* viewDims,
                                               uint64_t viewDimsNum,
                                               aclDataType dataType,
                                               const int64_t* stride,
                                               int64_t offset,
                                               aclFormat format,
                                               const int64_t* storageDims,
                                               uint64_t storageDimsNum,
                                               void* tensorData);
using _aclCreateScalarFuncPtr = aclScalar* (*)(void* value, aclDataType dataType);
using _aclCreateIntArrayFuncPtr = aclIntArray* (*)(const int64_t* value, uint64_t size);
using _aclCreateFloatArrayFuncPtr = aclFloatArray* (*)(const float* value, uint64_t size);
using _aclCreateBoolArrayFuncPtr = aclBoolArray* (*)(const bool* value, uint64_t size);
using _aclCreateTensorListFuncPtr = aclTensorList* (*)(const aclTensor* const* value, uint64_t size);

// Base acl deleters
using _aclDestroyTensorFuncPtr = int (*)(const aclTensor* tensor);
using _aclDestroyScalarFuncPtr = int (*)(const aclScalar* scalar);
using _aclDestroyIntArrayFuncPtr = int (*)(const aclIntArray* array);
using _aclDestroyFloatArrayFuncPtr = int (*)(const aclFloatArray* array);
using _aclDestroyBoolArrayFuncPtr = int (*)(const aclBoolArray* array);
using _aclDestroyTensorListFuncPtr = int (*)(const aclTensorList* array);
using _aclDestroyAclOpExecutorFuncPtr = int (*)(aclOpExecutor* executor);

// Init and finalize
using _aclnnInitFuncPtr = int (*)(const char*);
using _aclnnFinalizeFuncPtr = int (*)();

// For reusing aclOpExecutor
using _aclSetAclOpExecutorRepeatableFuncPtr = int (*)(aclOpExecutor* executor);

// Set the device address ptr for aclTensor
using _aclSetTensorAddrFuncPtr = int (*)(aclOpExecutor* executor, const size_t index, aclTensor* tensor, void* addr);
using _aclSetDynamicTensorAddrFuncPtr = int (*)(
    aclOpExecutor* executor,
    const size_t index,
    const size_t relativeIndex,
    aclTensorList* tensors,
    void* addr);

#define DECLARE_ACLNN_COMMON_META_FUNC(name) DA_API _##name##FuncPtr name##_ = nullptr

#define EXTERN_ACLNN_COMMON_META_FUNC(name) \
  extern _##name##FuncPtr name##_;          \
  inline constexpr const char* kName##name##_ = #name

EXTERN_ACLNN_COMMON_META_FUNC(aclCreateTensor);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateScalar);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateIntArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateFloatArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateBoolArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateTensorList);

EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyTensor);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyScalar);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyIntArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyFloatArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyBoolArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyTensorList);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyAclOpExecutor);

EXTERN_ACLNN_COMMON_META_FUNC(aclnnInit);
EXTERN_ACLNN_COMMON_META_FUNC(aclnnFinalize);

EXTERN_ACLNN_COMMON_META_FUNC(aclSetAclOpExecutorRepeatable);

EXTERN_ACLNN_COMMON_META_FUNC(aclSetTensorAddr);
EXTERN_ACLNN_COMMON_META_FUNC(aclSetDynamicTensorAddr);

#define GET_ACLNN_COMMON_META_FUNC(name) \
  []() -> auto {                         \
    if (name##_ == nullptr) {            \
      LoadOpApiLib();                    \
    }                                    \
    return name##_;                      \
  }()

#define GET_ACLNN_OP_FUNC(name) GetAclnnOpApiFunc(name.c_str())

} // namespace ops
} // namespace fxrt
#endif // __OPS_ASCEND_ACLNN_UTILS_ACLNN_COMMON_META_H__
