/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_ACL_OP_API_H
#define OP_API_ACL_OP_API_H

#include <acl/acl_base.h>
#include <stdint.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;

typedef int aclnnStatus;

constexpr aclnnStatus OK = 0;
constexpr aclnnStatus ERROR = 1;

enum aclnnTensorPlacement {
    ACLNN_MEMTYPE_DEVICE,                   ///< Tensor位于Device上的HBM内存
    ACLNN_MEMTYPE_HOST,                     ///< Tensor位于Host
    ACLNN_MEMTYPE_HOST_COMPILE_INDEPENDENT, ///< Tensor位于Host，且数据紧跟在结构体后面
    ACLNN_MEMTYPE_END
};

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY _declspec(dllexport)
#else
#define ACL_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define ACL_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define ACL_FUNC_VISIBILITY
#endif
#endif

#ifdef __GNUC__
#define ACL_DEPRECATED __attribute__((deprecated))
#define ACL_DEPRECATED_MESSAGE(message) __attribute__((deprecated(message)))
#elif defined(_MSC_VER)
#define ACL_DEPRECATED __declspec(deprecated)
#define ACL_DEPRECATED_MESSAGE(message) __declspec(deprecated(message))
#else
#define ACL_DEPRECATED
#define ACL_DEPRECATED_MESSAGE(message)
#endif

ACL_FUNC_VISIBILITY aclTensor *aclCreateTensor(const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
                                               const int64_t *stride, int64_t offset, aclFormat format,
                                               const int64_t *storage_dims, uint64_t storage_dims_num,
                                               void *tensor_data);

ACL_FUNC_VISIBILITY aclScalar *aclCreateScalar(void *value, aclDataType data_type);
ACL_FUNC_VISIBILITY aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size);
ACL_FUNC_VISIBILITY aclFloatArray *aclCreateFloatArray(const float *value, uint64_t size);
ACL_FUNC_VISIBILITY aclBoolArray *aclCreateBoolArray(const bool *value, uint64_t size);
ACL_FUNC_VISIBILITY aclTensorList *aclCreateTensorList(const aclTensor *const *value, uint64_t size);

ACL_FUNC_VISIBILITY aclnnStatus aclDestroyTensor(const aclTensor *tensor);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyScalar(const aclScalar *scalar);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyIntArray(const aclIntArray *array);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyFloatArray(const aclFloatArray *array);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyBoolArray(const aclBoolArray *array);
ACL_FUNC_VISIBILITY aclnnStatus aclDestroyTensorList(const aclTensorList *array);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACL_OP_API_H
