#include <ATen/Functions.h>

#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/aten/common/DLConvertor.h"

using namespace std;
namespace at {

DLDataType getDLDataType(const Tensor& t)
{
    DLDataType dtype;
    dtype.lanes = 1;
    // Convert element size to bits by multiplying with bits per byte
    constexpr int BITS_PER_BYTE = 8;
    dtype.bits = t.element_size() * BITS_PER_BYTE;
    switch (t.scalar_type()) {
        case ScalarType::UInt1:
        case ScalarType::UInt2:
        case ScalarType::UInt3:
        case ScalarType::UInt4:
        case ScalarType::UInt5:
        case ScalarType::UInt6:
        case ScalarType::UInt7:
        case ScalarType::Byte:
        case ScalarType::UInt16:
        case ScalarType::UInt32:
        case ScalarType::UInt64:
            dtype.code = DLDataTypeCode::kDLUInt;
            break;
        case ScalarType::Int1:
        case ScalarType::Int2:
        case ScalarType::Int3:
        case ScalarType::Int4:
        case ScalarType::Int5:
        case ScalarType::Int6:
        case ScalarType::Int7:
        case ScalarType::Char:
            dtype.code = DLDataTypeCode::kDLInt;
            break;
        // NOLINTNEXTLINE(bugprone-branch-clone)
        case ScalarType::Double:
            dtype.code = DLDataTypeCode::kDLFloat;
            break;
        case ScalarType::Float:
            dtype.code = DLDataTypeCode::kDLFloat;
            break;
        // NOLINTNEXTLINE(bugprone-branch-clone)
        case ScalarType::Int:
            dtype.code = DLDataTypeCode::kDLInt;
            break;
        case ScalarType::Long:
            dtype.code = DLDataTypeCode::kDLInt;
            break;
        case ScalarType::Short:
            dtype.code = DLDataTypeCode::kDLInt;
            break;
        case ScalarType::Half:
            dtype.code = DLDataTypeCode::kDLFloat;
            break;
        case ScalarType::Bool:
            dtype.code = DLDataTypeCode::kDLBool;
            break;
        case ScalarType::ComplexHalf:
        case ScalarType::ComplexFloat:
        case ScalarType::ComplexDouble:
            dtype.code = DLDataTypeCode::kDLComplex;
            break;
        case ScalarType::BFloat16:
            dtype.code = DLDataTypeCode::kDLBfloat;
            break;
        case ScalarType::Float8_e5m2:
        case ScalarType::Float8_e5m2fnuz:
        case ScalarType::Float8_e4m3fn:
        case ScalarType::Float8_e4m3fnuz:
        case ScalarType::Float8_e8m0fnu:
            TORCH_CHECK(false, "float8 types are not supported by dlpack", PTA_ERROR(ErrCode::TYPE));
            break;
        case ScalarType::QInt8:
        case ScalarType::QUInt8:
        case ScalarType::QInt32:
        case ScalarType::QUInt4x2:
        case ScalarType::QUInt2x4:
            TORCH_CHECK(false, "QUInt/QInt types are not supported by dlpack", PTA_ERROR(ErrCode::TYPE));
            break;
        case ScalarType::Bits1x8:
        case ScalarType::Bits2x4:
        case ScalarType::Bits4x2:
        case ScalarType::Bits8:
        case ScalarType::Bits16:
            TORCH_CHECK(false, "Bit types are not supported by dlpack", PTA_ERROR(ErrCode::TYPE));
            break;
        case ScalarType::Undefined:
            TORCH_CHECK(false, "Undefined is not a valid ScalarType", PTA_ERROR(ErrCode::TYPE));
        case ScalarType::NumOptions:
            TORCH_CHECK(false, "NumOptions is not a valid ScalarType", PTA_ERROR(ErrCode::TYPE));
    }
    return dtype;
}

static DLDevice getDLDevice(const Tensor& tensor, c10::DeviceIndex device_id)
{
    DLDevice ctx;
    ctx.device_id = static_cast<int32_t>(static_cast<unsigned char>(device_id));
    switch (tensor.device().type()) {
        case DeviceType::CPU:
            ctx.device_type = DLDeviceType::kDLCPU;
            break;
        case DeviceType::PrivateUse1:
            ctx.device_type = DLDeviceType::kDLExtDev;
            break;
        default:
            TORCH_CHECK(false, "Cannot pack tensors on " + tensor.device().str(), PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return ctx;
}

static Device getATenDevice(const DLDevice& ctx, void* data)
{
    switch (ctx.device_type) {
        case DLDeviceType::kDLCPU:
            return at::Device(DeviceType::CPU);
        case DLDeviceType::kDLExtDev:
            return at::Device(DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(ctx.device_id));
        default:
            TORCH_CHECK(false, "Unsupported device_type: ", std::to_string(ctx.device_type), PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
}

namespace {
constexpr int k8Bits = 8;
constexpr int k16Bits = 16;
constexpr int k32Bits = 32;
constexpr int k64Bits = 64;
constexpr int k128Bits = 128;
} // namespace

ScalarType toScalarType(const DLDataType& dtype)
{
    ScalarType stype = ScalarType::Undefined;
    TORCH_CHECK(dtype.lanes == 1, "ATen does not support lanes != 1", PTA_ERROR(ErrCode::NOT_SUPPORT));
    switch (dtype.code) {
        case DLDataTypeCode::kDLUInt:
            switch (dtype.bits) {
                case k8Bits:
                    stype = ScalarType::Byte;
                    break;
                case k16Bits:
                    stype = ScalarType::UInt16;
                    break;
                case k32Bits:
                    stype = ScalarType::UInt32;
                    break;
                case k64Bits:
                    stype = ScalarType::UInt64;
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported kUInt bits ", std::to_string(dtype.bits), PTA_ERROR(ErrCode::NOT_SUPPORT));
            }
            break;
        case DLDataTypeCode::kDLInt:
            switch (dtype.bits) {
                case k8Bits:
                    stype = ScalarType::Char;
                    break;
                case k16Bits:
                    stype = ScalarType::Short;
                    break;
                case k32Bits:
                    stype = ScalarType::Int;
                    break;
                case k64Bits:
                    stype = ScalarType::Long;
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported kInt bits ", std::to_string(dtype.bits), PTA_ERROR(ErrCode::NOT_SUPPORT));
            }
            break;
        case DLDataTypeCode::kDLFloat:
            switch (dtype.bits) {
                case k16Bits:
                    stype = ScalarType::Half;
                    break;
                case k32Bits:
                    stype = ScalarType::Float;
                    break;
                case k64Bits:
                    stype = ScalarType::Double;
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported kFloat bits ", std::to_string(dtype.bits), PTA_ERROR(ErrCode::NOT_SUPPORT));
            }
            break;
        case DLDataTypeCode::kDLBfloat:
            switch (dtype.bits) {
                case k16Bits:
                    stype = ScalarType::BFloat16;
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported kFloat bits ", std::to_string(dtype.bits), PTA_ERROR(ErrCode::NOT_SUPPORT));
            }
            break;
        case DLDataTypeCode::kDLComplex:
            switch (dtype.bits) {
                case k32Bits:
                    stype = ScalarType::ComplexHalf;
                    break;
                case k64Bits:
                    stype = ScalarType::ComplexFloat;
                    break;
                case k128Bits:
                    stype = ScalarType::ComplexDouble;
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported kFloat bits ", std::to_string(dtype.bits), PTA_ERROR(ErrCode::NOT_SUPPORT));
            }
            break;
        case DLDataTypeCode::kDLBool:
            switch (dtype.bits) {
                case k8Bits:
                    stype = ScalarType::Bool;
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported kDLBool bits ", std::to_string(dtype.bits), PTA_ERROR(ErrCode::NOT_SUPPORT));
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported code ", std::to_string(dtype.code), PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    return stype;
}

namespace {
struct ATenDLMTensor {
    Tensor handle;
    DLManagedTensor tensor{};
};
} // namespace

static void deleter(DLManagedTensor* arg)
{
    delete static_cast<ATenDLMTensor*>(arg->manager_ctx);
}

// This function returns a shared_ptr to memory managed DLpack tensor
// constructed out of ATen tensor
DLManagedTensor* toDLPack(const Tensor& src)
{
    // create a new tensor with possibly normalized strides
    // gh-83069
    auto shape = src.sizes();
    auto strides = src.strides().vec();
    static constexpr int64_t kMinDimForStride = 2;
    for (int i = 0; i < src.dim(); i++) {
        if (shape[i] < kMinDimForStride) {
            strides[i] = 1;
        }
    }

    auto view = src.as_strided(shape, strides, src.storage_offset());
    ATenDLMTensor* atDLMTensor(new ATenDLMTensor);
    atDLMTensor->handle = view;
    atDLMTensor->tensor.manager_ctx = atDLMTensor;
    atDLMTensor->tensor.deleter = &deleter;
    atDLMTensor->tensor.dl_tensor.data = view.data_ptr();
    c10::DeviceIndex device_id = 0;
    if (src.is_cuda() || src.is_privateuseone()) {
        device_id = src.get_device();
    }
    atDLMTensor->tensor.dl_tensor.device = getDLDevice(src, device_id);
    atDLMTensor->tensor.dl_tensor.ndim = static_cast<int32_t>(src.dim());
    atDLMTensor->tensor.dl_tensor.dtype = getDLDataType(src);
    atDLMTensor->tensor.dl_tensor.shape = view.sizes().data();
    atDLMTensor->tensor.dl_tensor.strides = view.strides().data();
    atDLMTensor->tensor.dl_tensor.byte_offset = 0;
    return &(atDLMTensor->tensor);
}

Tensor fromDLPack(DLManagedTensor* src)
{
    auto deleter = [src](void* self [[maybe_unused]]) {
        if (src->deleter) {
            src->deleter(src);
        }
    };
    return fromDLPack(src, std::move(deleter));
}

Tensor fromDLPack(DLManagedTensor* src, std::function<void(void*)> deleter)
{
    Device device = getATenDevice(src->dl_tensor.device, src->dl_tensor.data);
    ScalarType stype = toScalarType(src->dl_tensor.dtype);
    if (!src->dl_tensor.strides) {
        return at_npu::native::from_blob(
            src->dl_tensor.data,
            IntArrayRef(src->dl_tensor.shape, src->dl_tensor.ndim),
            std::move(deleter),
            at::device(device).dtype(stype),
            {device});
    }
    return at_npu::native::from_blob(
        src->dl_tensor.data,
        IntArrayRef(src->dl_tensor.shape, src->dl_tensor.ndim),
        IntArrayRef(src->dl_tensor.strides, src->dl_tensor.ndim),
        deleter,
        at::device(device).dtype(stype),
        {device});
}
} // namespace at
