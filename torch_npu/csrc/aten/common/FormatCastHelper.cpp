#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/aten/common/FormatCastHelper.h"

namespace at_npu {
namespace native {

bool FormatCastHelper::IsSameGroupType(const at::Tensor& src, const at::Tensor& dst)
{
    auto src_format = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_.npu_format_;
    auto dst_format = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_.npu_format_;
    return FormatHelper::GetBaseFormat(src_format) == FormatHelper::GetBaseFormat(dst_format);
}

void FormatCastHelper::base_format_cast_nocheck(at::Tensor& dst, const at::Tensor& src)
{
    dst.set_(dst.storage(), src.storage_offset(), src.sizes(), src.strides());
    NPUNativeFunctions::copy_memory_(dst, src, true);
}

void FormatCastHelper::format_cast_as_base_format(const at::Tensor& src, aclFormat format)
{
    AT_ASSERT(FormatHelper::IsBaseFormatType(format), "dst format must be base format", PTA_ERROR(ErrCode::PARAM));
    AT_ASSERT(FormatHelper::IsBaseFormatType(src), "src format must be base format", PTA_ERROR(ErrCode::PARAM));

    auto& src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    // due to CANN principle : if the ori format of a tensor is the
    // same as the npu format, then its base shape must be same as storage shape
    // so we should not change the storage shape when format cast between base format
    src_desc.origin_format_ = format;
    src_desc.npu_format_ = format;
    return;
}

bool FormatCastHelper::format_cast_between_group(
    at::Tensor& dst,
    const at::Tensor& src,
    FormatCastHelper::FormatCastFunc format_cast_inside_group)
{
    if (FormatHelper::IsBaseFormatType(src)) {
        if (FormatHelper::IsBaseFormatType(dst)) {
            // src base format (src format) -> dst base format
            base_format_cast_nocheck(dst, src); // only need to copy memory
            return true;
        } else {
            // src base format (src format) -> dst base format
            // dst base format -> dst format
            auto src_base_format = FormatHelper::GetBaseFormat(src);
            format_cast_as_base_format(src, FormatHelper::GetBaseFormat(dst)); // prepare: covert src to dst base format
            format_cast_inside_group(dst, src); // src base format (src format) -> dst base format
            format_cast_as_base_format(src, src_base_format); // recover: dst base format -> dst format
            return true;
        }
    } else {
        if (FormatHelper::IsBaseFormatType(dst)) {
            // src format -> src base format
            // src base format -> dst base format (dst format)
            auto dst_base_format = FormatHelper::GetBaseFormat(dst);
            format_cast_as_base_format(dst, FormatHelper::GetBaseFormat(src)); // prepare: cover dst to src base format
            format_cast_inside_group(dst, src); // src format -> src base format
            format_cast_as_base_format(dst, dst_base_format); // recover: src base format -> dst format
            return true;
        }
    }
    return false;
}

at::Tensor FormatCastHelper::ApplyBaseFormatTensorBy(const at::Tensor& src)
{
    auto format = FormatHelper::GetBaseFormat(src);
    return custom_ops::npu_format_cast(src, format);
}

at::Tensor& FormatCastHelper::CovertSelfToBaseFormat(at::Tensor& src)
{
    auto format = FormatHelper::GetBaseFormat(src);
    return NPUNativeFunctions::npu_format_cast_(src, format);
}

} // namespace native
} // namespace at_npu
