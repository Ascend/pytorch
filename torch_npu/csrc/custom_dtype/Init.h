#pragma once

#include <ATen/ATen.h>
#ifndef BUILD_LIBTORCH
#include <torch/csrc/python_headers.h>
#endif
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "third_party/acl/inc/acl/acl_base.h"

namespace c10_npu {
const int g_toAclOffset = 256;

#define ENUM_OFFSET(new_name, old_name) new_name = static_cast<int>(old_name) + g_toAclOffset,

#ifndef BUILD_LIBTORCH
TORCH_NPU_API PyMethodDef* custom_dtype_functions();
#endif

enum class DType {
    UNDEFINED = -1,
    ENUM_OFFSET(FLOAT, ACL_FLOAT)
    ENUM_OFFSET(FLOAT16, ACL_FLOAT16)
    ENUM_OFFSET(INT8, ACL_INT8)
    ENUM_OFFSET(INT32, ACL_INT32)
    ENUM_OFFSET(UINT8, ACL_UINT8)
    ENUM_OFFSET(INT16, ACL_INT16)
    ENUM_OFFSET(UINT16, ACL_UINT16)
    ENUM_OFFSET(UINT32, ACL_UINT32)
    ENUM_OFFSET(INT64, ACL_INT64)
    ENUM_OFFSET(UINT64, ACL_UINT64)
    ENUM_OFFSET(DOUBLE, ACL_DOUBLE)
    ENUM_OFFSET(BOOL, ACL_BOOL)
    ENUM_OFFSET(STRING, ACL_STRING)
    ENUM_OFFSET(COMPLEX64, ACL_COMPLEX64)
    ENUM_OFFSET(COMPLEX128, ACL_COMPLEX128)
    ENUM_OFFSET(BF16, ACL_BF16)
    ENUM_OFFSET(INT4, ACL_INT4)
    ENUM_OFFSET(UINT1, ACL_UINT1)
    ENUM_OFFSET(COMPLEX32, ACL_COMPLEX32)
};

inline bool IsCustomDType(int64_t t)
{
    if (t >= g_toAclOffset) {
        return true;
    }
    return false;
}

// Both c10_npu::DType and ScalarType are supported
inline aclDataType GetAclDataType(int64_t t)
{
    if (t >= g_toAclOffset) {
        return static_cast<aclDataType>(t - g_toAclOffset);
    }
    return at_npu::native::OpPreparation::convert_to_acl_data_type(
        static_cast<at::ScalarType>(t));
}

inline aclDataType GetAclDataType(DType t)
{
    return static_cast<aclDataType>(static_cast<int32_t>(t) - g_toAclOffset);
}

inline at::ScalarType GetATenDType(int64_t t)
{
    aclDataType aclType = GetAclDataType(t);
    return at_npu::native::OpPreparation::convert_to_scalar_type(aclType);
}

const std::string CustomDataTypeToString(int64_t dType);

} // namespace c10_npu
