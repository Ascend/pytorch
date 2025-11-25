#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/NpuStorageOffsetGuard.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/aten/common/FormatCastHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/custom_dtype/Init.h"
#include "third_party/op-plugin/op_plugin/utils/op_api_common.h"

namespace at_npu {
namespace native {

using tensor_list = std::vector<at::Tensor>;
using GetFormatFunc = int (*)(const aclTensor *, const int, const int, int64_t **, uint64_t *, int *);

std::tuple<bool, int64_t, c10::SmallVector<int64_t, SIZE>> MaybeUseAclnnNpuFormatCast(const at::Tensor& src,
    int64_t acl_format, c10::optional<int64_t> customize_dtype)
{
    const static auto GetFormatFuncAddr = GetOpApiFuncAddr("aclnnNpuFormatCastCalculateSizeAndFormat");
    const static auto FormatCastFuncAddr = GetOpApiFuncAddr("aclnnNpuFormatCast");

    const static bool aclnnNpuFormatCastExist =
        (GetFormatFuncAddr == nullptr || FormatCastFuncAddr == nullptr) ? false : true;

    GetFormatFunc GetFormat = reinterpret_cast<GetFormatFunc>(GetFormatFuncAddr);
    int64_t *dstStorageShape = nullptr;
    uint64_t dstShapeSize = 0;
    int dstFormat;
    at::SmallVector<int64_t, SIZE> outputShape = {};
    aclDataType customizeAcltype = (customize_dtype.has_value()) ?
        c10_npu::GetAclDataType(customize_dtype.value()) :
        at_npu::native::OpPreparation::convert_to_acl_data_type(src.scalar_type());

    if (c10_npu::IsAclnnOnly()) {
        if (aclnnNpuFormatCastExist) {
            auto api_ret = GetFormat(ConvertType(src), acl_format, customizeAcltype, &dstStorageShape,
                &dstShapeSize, &dstFormat);
            NPU_CHECK_ERROR(api_ret, "aclnnNpuFormatCastCalculateSizeAndFormat");
            for (uint64_t i = 0; i < dstShapeSize; i++) {
                outputShape.push_back(dstStorageShape[i]);
            }
            delete[] dstStorageShape;
            return std::make_tuple(true, dstFormat, outputShape);
        }
        TORCH_CHECK(false,
            "aclnnNpuFormatCast does not exist, Current device only support aclnn operators.",
            PTA_ERROR(ErrCode::NOT_SUPPORT));
    }
    if (at_npu::native::env::CheckJitDisable()) {
        if (aclnnNpuFormatCastExist) {
            auto api_ret = GetFormat(ConvertType(src), acl_format, customizeAcltype, &dstStorageShape,
                &dstShapeSize, &dstFormat);
            if (api_ret != 0) {
                if (customize_dtype.has_value()) {
                    NPU_CHECK_ERROR(api_ret, "aclnnNpuFormatCastCalculateSizeAndFormat");
                }
                return std::make_tuple(false, dstFormat, outputShape);
            }
            for (uint64_t i = 0; i < dstShapeSize; i++) {
                outputShape.push_back(dstStorageShape[i]);
            }
            delete[] dstStorageShape;
            return std::make_tuple(true, dstFormat, outputShape);
        } else {
            if (C10_UNLIKELY(customize_dtype.has_value())) {
                TORCH_CHECK(false,
                    "customize_dtype is not supported while aclnnNpuFormatCast does not exist.",
                    PTA_ERROR(ErrCode::NOT_SUPPORT));
            }
            return std::make_tuple(false, dstFormat, outputShape);
        }
    } else {
        if (C10_UNLIKELY(customize_dtype.has_value())) {
            TORCH_CHECK(false,
                "customize_dtype is not supported while jit_compile=True.",
                PTA_ERROR(ErrCode::NOT_SUPPORT));
        }
        return std::make_tuple(false, dstFormat, outputShape);
    }
}

at::Tensor create_tensor_with_format_and_shape(c10::IntArrayRef baseSizes,
    c10::IntArrayRef storageSizes,
    const caffe2::TypeMeta dtype, int64_t acl_format)
{
    c10::Allocator *allocator = c10_npu::NPUCachingAllocator::get();
    int64_t nelements = 1;
    for (const auto& num : storageSizes) {
        nelements *= num;
    }
    int64_t size_bytes = nelements * dtype.itemsize();
    c10::intrusive_ptr<c10::StorageImpl> storage_impl = torch_npu::make_npu_storage_impl(
        c10::StorageImpl::use_byte_size_t(),
        c10::SymInt(size_bytes),
        allocator->allocate(size_bytes),
        allocator,
        true);
    auto tensor = at::detail::make_tensor<torch_npu::NPUTensorImpl>(storage_impl, dtype);

    if (baseSizes.size() != 1 || baseSizes[0] != 0) {
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous(baseSizes);
    }
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(c10::MemoryFormat::Contiguous);
    StorageDescHelper::SetDesc(tensor, baseSizes, storageSizes, tensor.strides(), static_cast<aclFormat>(acl_format));
    return tensor;
}

at::Tensor format_cast_impl_out_npu_aclnn(const at::Tensor& src,
    int64_t acl_format, c10::IntArrayRef storageSizes)
{
    auto src_new = src.contiguous();
    auto src_new_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src_new)->npu_desc_;

    at::Tensor dst = create_tensor_with_format_and_shape(
        src_new.sizes(), storageSizes, src.dtype(), acl_format);

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnNpuFormatCast, src_new, dst);

    // format cast only change physical layout of base tensor and view tensor's
    // metadata remain unchanged
    dst.set_(dst.storage(), src_new.storage_offset(), src_new.sizes(), src_new.strides());
    return dst;
}

at::Tensor format_cast_impl_out_npu(at::Tensor& dst, const at::Tensor& src)
{
    string srcFormat = FormatHelper::GetFormatName(src);
    string dstFormat = FormatHelper::GetFormatName(dst);

    if (!FormatCastHelper::IsSameGroupType(src, dst)) {
        bool res = FormatCastHelper::format_cast_between_group(dst, src, format_cast_impl_out_npu);
        if (!res) {
            AT_ERROR("unsupport cast from ", srcFormat, " to ", dstFormat);
        }
        return dst;
    }

    NpuStorageOffsetGuard guard_input(const_cast<at::Tensor &>(src));
    NpuStorageOffsetGuard guard_output(dst);
    OpCommand cmd;
    cmd.Name("Identity")
       .InputWithoutContiguous(src)
       .Output(dst)
       .Run();
    return dst;
}

// convert src from src_format to dst_format, write the result into dst(self)
at::Tensor& NPUNativeFunctions::npu_format_cast_(at::Tensor& self, const at::Tensor& src,
                                                 c10::optional<int64_t> customize_dtype)
{
    torch_npu::utils::torch_check_npu(self);
    torch_npu::utils::torch_check_npu(src);
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    auto dst_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
    if (src_desc.npu_format_ == dst_desc.npu_format_) {
        self.copy_(src);
        return self;
    }

    auto [useAclnn, outFormat, StorageShape] = MaybeUseAclnnNpuFormatCast(self, dst_desc.npu_format_, customize_dtype);
    if (useAclnn == true) {
        at::Tensor dst = format_cast_impl_out_npu_aclnn(self, outFormat, StorageShape);
        self.set_(dst.storage(), dst.storage_offset(), dst.sizes(), dst.strides());
        return self;
    }

    // calculate the output result of the NPU
    format_cast_impl_out_npu(self, src);

    return self;
}

// conver self to acl_format, write the result into new result tensor
at::Tensor npu_format_cast_impl(
    const at::Tensor& src,
    int64_t acl_format)
{
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    at::Tensor dst = OpPreparation::ApplyTensorWithFormat(
        src_desc.base_sizes_, src.options(), acl_format);

    // calculate the output result of the NPU
    format_cast_impl_out_npu(dst, src);

    // format cast only change physical layout of base tensor and view tensor's
    // metadata remain unchanged
    dst.set_(dst.storage(), src.storage_offset(), src.sizes(), src.strides());
    return dst;
}

// conver self to dst'format, write the result into new result tensor
at::Tensor NPUNativeFunctions::npu_format_cast(
    const at::Tensor& self,
    const at::Tensor& dst,
    c10::optional<int64_t> customize_dtype)
{
    torch_npu::utils::torch_check_npu(dst);
    auto dst_desc = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_;
    int64_t dst_format = dst_desc.npu_format_;
    return custom_ops::npu_format_cast(self, dst_format, customize_dtype);
}

// conver self to acl_format, write the result into self
at::Tensor& NPUNativeFunctions::npu_format_cast_(
    at::Tensor& self,
    int64_t acl_format,
    c10::optional<int64_t> customize_dtype)
{
    torch_npu::utils::torch_check_npu(self);
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
    if (src_desc.npu_format_ == acl_format) {
        return self;
    }
    if (FormatHelper::IsBaseFormatType(self) &&
        FormatHelper::IsBaseFormatType(static_cast<aclFormat>(acl_format))) {
        FormatCastHelper::format_cast_as_base_format(self, static_cast<aclFormat>(acl_format));
        return self;
    }

    auto [useAclnn, outFormat, StorageShape] = MaybeUseAclnnNpuFormatCast(self, acl_format, customize_dtype);
    if (useAclnn == true) {
        at::Tensor dst = format_cast_impl_out_npu_aclnn(self, outFormat, StorageShape);
        self.set_(dst.storage(), dst.storage_offset(), dst.sizes(), dst.strides());
        return self;
    }

    at::Tensor dst = OpPreparation::ApplyTensorWithFormat(
        src_desc.base_sizes_, self.options(), acl_format);

    // calculate the output result of the NPU
    format_cast_impl_out_npu(dst, self);

    // format cast only change physical layout of base tensor and view tensor's
    // metadata remain unchanged
    self.set_(dst.storage(), self.storage_offset(), self.sizes(), self.strides());

    return self;
}

int64_t NPUNativeFunctions::get_npu_format(const at::Tensor& self)
{
    torch_npu::utils::torch_check_npu(self);
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
    return src_desc.npu_format_;
}

at::Tensor NPUNativeFunctions::_npu_format_cast(const at::Tensor& self, int64_t acl_format)
{
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
    if (src_desc.npu_format_ == acl_format) {
        ASCEND_LOGD("no need to do format cast");
        return self;
    }
    if (FormatHelper::IsBaseFormatType(self) &&
        FormatHelper::IsBaseFormatType(static_cast<aclFormat>(acl_format))) {
        FormatCastHelper::format_cast_as_base_format(self, static_cast<aclFormat>(acl_format));
        return self;
    }
    auto [useAclnn, outFormat, StorageShape] = MaybeUseAclnnNpuFormatCast(self, acl_format, c10::nullopt);
    if (useAclnn == false) {
        return npu_format_cast_impl(self, acl_format);
    }
    return format_cast_impl_out_npu_aclnn(self, outFormat, StorageShape);
}

at::Tensor NPUNativeFunctions::_npu_format_cast(const at::Tensor& self, int64_t acl_format,
                                                int64_t customize_dtype)
{
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
    if (src_desc.npu_format_ == acl_format) {
        ASCEND_LOGD("no need to do format cast");
        return self;
    }
    if (FormatHelper::IsBaseFormatType(self) &&
        FormatHelper::IsBaseFormatType(static_cast<aclFormat>(acl_format))) {
        FormatCastHelper::format_cast_as_base_format(self, static_cast<aclFormat>(acl_format));
        return self;
    }
    auto [useAclnn, outFormat, StorageShape] = MaybeUseAclnnNpuFormatCast(self, acl_format, customize_dtype);
    if (useAclnn == false) {
        return npu_format_cast_impl(self, acl_format);
    }
    return format_cast_impl_out_npu_aclnn(self, outFormat, StorageShape);
}

at::Tensor NPUNativeFunctions::npu_format_cast(const at::Tensor& self, int64_t acl_format,
                                               c10::optional<int64_t> customize_dtype)
{
    torch_npu::utils::torch_check_npu(self);
    if (NPUNativeFunctions::get_npu_format(self) == acl_format) {
        ASCEND_LOGD("no need to do format cast");
        return self;
    }
    if (customize_dtype.has_value()) {
        return custom_ops::_npu_format_cast(self, acl_format, customize_dtype.value());
    }
    return custom_ops::_npu_format_cast(self, acl_format);
}

} // namespace native
} // namespace at_npu
