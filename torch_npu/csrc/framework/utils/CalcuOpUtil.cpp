#include <ATen/record_function.h>

#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/aten/mirror/NPUMemoryOverlap.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/framework/contiguous/ReshapeOpt.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/ForceJitCompileList.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"

namespace {
constexpr float EPSILON = 1e-6;

// check all at::ScalarType is not negative
#define ENUM_PAIR_FUNC(_1, n) static_assert(static_cast<int64_t>(at::ScalarType::n) >= 0, #n " is negative");
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(ENUM_PAIR_FUNC)
#undef ENUM_PAIR_FUNC

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)                                                                    \
    _(at::ScalarType::Byte, ACL_UINT8)                                                                                 \
    _(at::ScalarType::Char, ACL_INT8)                                                                                  \
    _(at::ScalarType::Short, ACL_INT16)                                                                                \
    _(at::ScalarType::Int, ACL_INT32)                                                                                  \
    _(at::ScalarType::Long, ACL_INT64)                                                                                 \
    _(at::ScalarType::Half, ACL_FLOAT16)                                                                               \
    _(at::ScalarType::Float, ACL_FLOAT)                                                                                \
    _(at::ScalarType::Double, ACL_DOUBLE)                                                                              \
    _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)                                                                      \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)                                                                     \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)                                                                   \
    _(at::ScalarType::Bool, ACL_BOOL)                                                                                  \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::BFloat16, ACL_BF16)                                                                              \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)                                                                      \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)                                                                      \
    _(at::ScalarType::Bits1x8, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits2x4, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits4x2, ACL_DT_UNDEFINED)                                                                       \
    _(at::ScalarType::Bits8, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::Bits16, ACL_DT_UNDEFINED)                                                                        \
    _(at::ScalarType::Float8_e5m2, ACL_DT_UNDEFINED)                                                                   \
    _(at::ScalarType::Float8_e4m3fn, ACL_DT_UNDEFINED)                                                                 \
    _(at::ScalarType::Float8_e5m2fnuz, ACL_DT_UNDEFINED)                                                               \
    _(at::ScalarType::Float8_e4m3fnuz, ACL_DT_UNDEFINED)                                                               \
    _(at::ScalarType::UInt16, ACL_UINT16)                                                                              \
    _(at::ScalarType::UInt32, ACL_UINT32)                                                                              \
    _(at::ScalarType::UInt64, ACL_UINT64)                                                                              \
    _(at::ScalarType::UInt1, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt2, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt3, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt4, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt5, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt6, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::UInt7, ACL_DT_UNDEFINED)                                                                         \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)                                                                     \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// check at::ScalarType has been changed or not
#define ENUM_PAIR_FUNC(at_dtype, acl_dtype)                                                                            \
    static_assert(kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at_dtype)] == (acl_dtype),                    \
                  #at_dtype " and " #acl_dtype " is not match any more, please check "                                 \
                            "AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR and modify it");
AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(ENUM_PAIR_FUNC)
#undef DEFINE_ENUM

static std::map<const std::string, const aclDataType> STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP = {
    {"uint16", ACL_UINT16}, {"uint8", ACL_UINT8}, {"uint64", ACL_UINT64}, {"string", ACL_STRING}};

aclError AclrtMemcpyAsyncParamCheck(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind,
                                    aclrtStream stream)
{
    auto ret = aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
    return ret;
}

aclError AclrtMemcpyParamCheck(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    auto ret = aclrtMemcpy(dst, destMax, src, count, kind);
    return ret;
}
} // namespace

namespace at_npu {
namespace native {
aclDataType CalcuOpUtil::ConvertToAclDataType(const at::ScalarType &data_type)
{
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED, std::string(c10::toString(data_type)) + " has not been supported",
                OPS_ERROR(ErrCode::NOT_SUPPORT))
    return acl_dtype;
}

aclDataType CalcuOpUtil::ConvertToAclDataType(const at::ScalarType &data_type, const std::string &realDataType)
{
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED, std::string(c10::toString(data_type)) + " has not been supported",
                OPS_ERROR(ErrCode::NOT_SUPPORT))
    if (!realDataType.empty()) {
        return STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP[realDataType];
    }
    return acl_dtype;
}

c10::Scalar CalcuOpUtil::ConvertTensorToScalar(const at::Tensor &tensor)
{
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
    } else {
        ASCEND_LOGE("unsupport scalar type! ");
        NPU_CHECK_ERROR(ACL_ERROR_UNSUPPORTED_DATA_TYPE);
    }

    return expScalar;
}

at::Tensor CalcuOpUtil::CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type)
{
    return CalcuOpUtil::CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

at::Tensor CalcuOpUtil::CopyTensorHostToDevice(const at::Tensor &cpu_tensor)
{
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    NPU_CHECK_ERROR(c10_npu::GetDevice(&deviceIndex));
    return cpuPinMemTensor.to(c10::Device(c10::DeviceType::PrivateUse1, deviceIndex), cpuPinMemTensor.scalar_type(),
                              true, true);
}

NPUStatus CalcuOpUtil::AclrtMemcpyAsync(const std::pair<at::Tensor, int64_t> &dst, size_t dst_size,
                                        const std::pair<at::Tensor, int64_t> &src, size_t src_size,
                                        aclrtMemcpyKind kind)
{
    void *dst_ptr = reinterpret_cast<uint8_t *>(dst.first.data_ptr()) + dst.second * dst.first.itemsize();
    void *src_ptr = reinterpret_cast<uint8_t *>(src.first.data_ptr()) + src.second * src.first.itemsize();
    NPU_CHECK_ERROR(
        c10_npu::queue::LaunchAsyncCopyTask(dst_ptr, dst_size, const_cast<void *>(src_ptr), src_size, kind));

    return "SUCCESS";
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(const StorageAndOffsetMemSizePair &dst, size_t dstMax,
                                                const StorageAndOffsetMemSizePair &src, size_t count,
                                                aclrtMemcpyKind kind)
{
    void *dst_ptr = static_cast<void *>(static_cast<uint8_t *>(const_cast<void *>(dst.first->data())) + dst.second);
    void *src_ptr = static_cast<void *>(static_cast<uint8_t *>(const_cast<void *>(src.first->data())) + src.second);
    return AclrtMemcpyParamCheck(dst_ptr, dstMax, const_cast<void *>(src_ptr), count, kind);
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(const StorageAndOffsetMemSizePair &dst, size_t dstMax, const void *src,
                                                size_t count, aclrtMemcpyKind kind)
{
    void *dst_ptr = static_cast<void *>(static_cast<uint8_t *>(const_cast<void *>(dst.first->data())) + dst.second);
    return AclrtMemcpyParamCheck(dst_ptr, dstMax, src, count, kind);
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(void *dst, size_t dstMax, const StorageAndOffsetMemSizePair &src,
                                                size_t count, aclrtMemcpyKind kind)
{
    void *src_ptr = static_cast<void *>(static_cast<uint8_t *>(const_cast<void *>(src.first->data())) + src.second);
    return AclrtMemcpyParamCheck(dst, dstMax, const_cast<void *>(src_ptr), count, kind);
}

aclError CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(const at::Tensor &dst, size_t dstMax, const at::Tensor &src,
                                                        size_t count, aclrtMemcpyKind kind)
{
    aclError ret = c10_npu::queue::LaunchAsyncCopyTask(dst.data_ptr(), dstMax, src.data_ptr(), count, kind);
    return ret;
}

aclError CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(const c10::StorageImpl &dst, size_t dstMax, void *src,
                                                        size_t count, aclrtMemcpyKind kind)
{
    aclError ret = c10_npu::queue::LaunchAsyncCopyTask(const_cast<void *>(dst.data()), dstMax, src, count, kind);
    return ret;
}

int64_t CalcuOpUtil::GetTensorNpuFormat(const at::Tensor &tensor)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "Expected all tensors to be on the same device. "
                "Expected NPU tensor, please check whether the input tensor "
                "device is correct.",
                OPS_ERROR(ErrCode::TYPE));
    if (NpuUtils::check_match(&tensor) || NpuUtils::check_5d_5d_match(tensor)) {
        const torch_npu::NPUStorageDesc &tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
        return tensor_desc.npu_format_;
    } else if (tensor.data_ptr() == nullptr) {
        // transforming faketensor into realtensor and assigning format ND
        return ACL_FORMAT_ND;
    } else {
        return InferFormat::GuessFormatWhenContiguous(tensor);
    }
}

void CalcuOpUtil::CheckMemoryOverLaps(c10::ArrayRef<at::Tensor> inputs, c10::ArrayRef<at::Tensor> outputs)
{
    for (const auto i : c10::irange(outputs.size())) {
        if (!outputs[i].defined()) {
            continue;
        }

        assert_no_internal_overlap(outputs[i]);

        for (const auto j : c10::irange(inputs.size())) {
            assert_no_partial_overlap(outputs[i], inputs[j]);
        }
    }
}

bool CalcuOpUtil::IsScalarWrappedToTensor(const at::Tensor &tensor)
{
    return tensor.unsafeGetTensorImpl()->is_wrapped_number() && (!torch_npu::utils::is_npu(tensor));
}

float CalcuOpUtil::GetScalarFloatValue(const c10::Scalar &scalar)
{
    float value;
    if (scalar.isFloatingPoint()) {
        value = scalar.toFloat();
    } else {
        value = static_cast<float>(scalar.toInt());
    }

    return value;
}

c10::SmallVector<int64_t, SHAPE_SIZE> CalcuOpUtil::ConvertIntArrayRefToSmallVector(c10::IntArrayRef intArray)
{
    c10::SmallVector<int64_t, SHAPE_SIZE> intVec;
    for (const auto i : c10::irange(intArray.size())) {
        intVec.emplace_back(intArray[i]);
    }

    return intVec;
}

using aclCubeMathType = enum : int8_t {
    KEEP_DTYPE = 0,
    ALLOW_FP32_DOWN_PRECISION = 1,
    USE_FP16 = 2,
    USE_HF32 = 3,
};

static std::unordered_map<uint8_t, aclCubeMathType> ACL_CUBE_MATH_TYPE_MAP = {
    {0b00, KEEP_DTYPE}, {0b01, USE_FP16}, {0b10, USE_HF32}, {0b11, ALLOW_FP32_DOWN_PRECISION}};

int8_t CalcuOpUtil::GetCubeMathType(bool allowHf32)
{
    bool allowFp32ToFp16 = native::env::IsAllowFP32ToFP16();
    uint8_t CubeMathTypeCode = (static_cast<uint8_t>(allowHf32) << 1) + static_cast<uint8_t>(allowFp32ToFp16);
    auto iter = ACL_CUBE_MATH_TYPE_MAP.find(CubeMathTypeCode);
    if (iter == ACL_CUBE_MATH_TYPE_MAP.end()) {
        return ALLOW_FP32_DOWN_PRECISION;
    }
    return iter->second;
}

} // namespace native
} // namespace at_npu
