#ifndef FXRT_ACLNN_META_API_H
#define FXRT_ACLNN_META_API_H

#include <cstddef>
#include <cstdint>

#include "acl/acl_base.h"

namespace fxrt {
namespace ops {

using aclnnStatus = std::int32_t;
inline constexpr aclnnStatus kAclnnSuccess = 0;

// Opaque ACLNN handles. Keep these types in their original runtime namespace.
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;

// Shared C interface types only; loading and cached addresses belong to callers.
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

using _aclDestroyTensorFuncPtr = aclnnStatus (*)(const aclTensor* tensor);
using _aclDestroyScalarFuncPtr = aclnnStatus (*)(const aclScalar* scalar);
using _aclDestroyIntArrayFuncPtr = aclnnStatus (*)(const aclIntArray* array);
using _aclDestroyFloatArrayFuncPtr = aclnnStatus (*)(const aclFloatArray* array);
using _aclDestroyBoolArrayFuncPtr = aclnnStatus (*)(const aclBoolArray* array);
using _aclDestroyTensorListFuncPtr = aclnnStatus (*)(const aclTensorList* array);
using _aclDestroyAclOpExecutorFuncPtr = aclnnStatus (*)(aclOpExecutor* executor);

using _aclnnInitFuncPtr = aclnnStatus (*)(const char*);
using _aclnnFinalizeFuncPtr = aclnnStatus (*)();
using _aclSetAclOpExecutorRepeatableFuncPtr = aclnnStatus (*)(aclOpExecutor* executor);
using _aclSetTensorAddrFuncPtr =
    aclnnStatus (*)(aclOpExecutor* executor, const size_t index, aclTensor* tensor, void* addr);
using _aclSetDynamicTensorAddrFuncPtr = aclnnStatus (*)(
    aclOpExecutor* executor,
    const size_t index,
    const size_t relativeIndex,
    aclTensorList* tensors,
    void* addr);

} // namespace ops
} // namespace fxrt

#endif // FXRT_ACLNN_META_API_H
