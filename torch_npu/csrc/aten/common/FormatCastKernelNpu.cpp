#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/NpuStorageOffsetGuard.h"
#include "torch_npu/csrc/aten/common/FormatCastHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace at_npu {
namespace native {

using tensor_list = std::vector<at::Tensor>;

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
at::Tensor& NPUNativeFunctions::npu_format_cast_(at::Tensor& self, const at::Tensor& src)
{
    torch_npu::utils::torch_check_npu(self);
    torch_npu::utils::torch_check_npu(src);
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    auto dst_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
    if (src_desc.npu_format_ == dst_desc.npu_format_) {
        self.copy_(src);
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
    if (src_desc.npu_format_ == acl_format) {
        ASCEND_LOGD("no need to do format cast");
        return src;
    }
    if (FormatHelper::IsBaseFormatType(src) &&
        FormatHelper::IsBaseFormatType(static_cast<aclFormat>(acl_format))) {
        FormatCastHelper::format_cast_as_base_format(src, static_cast<aclFormat>(acl_format));
        return src;
    }

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
    const at::Tensor& dst)
{
    torch_npu::utils::torch_check_npu(dst);
    auto dst_desc = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_;
    int64_t dst_format = dst_desc.npu_format_;
    return custom_ops::npu_format_cast(self, dst_format);
}

// conver self to acl_format, write the result into self
at::Tensor& NPUNativeFunctions::npu_format_cast_(
    at::Tensor& self,
    int64_t acl_format)
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
    return npu_format_cast_impl(self, acl_format);
}

at::Tensor NPUNativeFunctions::npu_format_cast(const at::Tensor& self, int64_t acl_format)
{
    torch_npu::utils::torch_check_npu(self);
    if (NPUNativeFunctions::get_npu_format(self) == acl_format) {
        ASCEND_LOGD("no need to do format cast");
        return self;
    }
    return custom_ops::_npu_format_cast(self, acl_format);
}

} // namespace native
} // namespace at_npu
