#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/framework/InferFormat.h"

namespace at_npu {
namespace native {

aclFormat InferFormat::GuessFormatWhenContiguous(const at::Tensor &tensor)
{
    // fix: when input tensor is a FakeTensor without desc.
    auto tensor_storage_impl = torch_npu::NPUBridge::GetNpuStorageImpl(tensor);
    if (tensor_storage_impl->data_ptr() == nullptr) {
        return ACL_FORMAT_ND;
    }
    auto desc = tensor_storage_impl->npu_desc_;
    // fix: NCDHW -> default format
    if ((desc.origin_format_ == ACL_FORMAT_NCDHW)) {
        if ((tensor.sizes().size() != desc.base_sizes_.size()) && (tensor.sizes().size() <= 4)) {
            return ACL_FORMAT_NCHW;
        }
    }
    return desc.origin_format_;
}

// NOTE: this method should cooperate with shape infer.
std::tuple<aclFormat, aclFormat> InferFormat::GuessFormatUnit(const c10::IntArrayRef &size, aclFormat format)
{
    aclFormat baseFormat = FormatHelper::GetBaseFormat(format);
    if ((baseFormat == ACL_FORMAT_NCDHW) && (size.size() > 4)) {
        return std::make_tuple(ACL_FORMAT_NCDHW, format);
    } else if (format == ACL_FORMAT_ND && size.size() == 4) {
        // 4 dim tensor must be NCHW, reflush base format
        return std::make_tuple(ACL_FORMAT_NCHW, ACL_FORMAT_NCHW);
    } else {
        if (baseFormat == ACL_FORMAT_NCDHW) {
            // scence: Dimensionality reduction: NCDHW->NCHW, for example: max/min
            // NOTE(NPU Dimensionality reduction)
            if (size.size() == 4) {
                return std::make_tuple(ACL_FORMAT_NCHW, ACL_FORMAT_NCHW);
            }
        }
    }
    return std::make_tuple(baseFormat, format);
}

aclFormat InferFormat::GuessBaseFormat(const c10::IntArrayRef &size)
{
    if (size.size() == 5) {
        return ACL_FORMAT_NCDHW;
    } else if (size.size() == 4) {
        return ACL_FORMAT_NCHW;
    }
    return ACL_FORMAT_ND;
}

aclFormat InferFormat::GuessStorageFormat(const c10::IntArrayRef &size, aclFormat format)
{
    if (format == ACL_FORMAT_FRACTAL_NZ && size.size() < 2) {
        // scalar scene and rank=1 scene do not support NZ
        TORCH_WARN_ONCE("Cannot create tensor with NZ format while dim < 2, "
                        "tensor will be created with ND format.");
        return ACL_FORMAT_ND;
    }

    int64_t dim = static_cast<int64_t>(size.size());
    aclFormat baseFormat = FormatHelper::GetBaseFormat(format);
    bool isBaseFormat = (baseFormat == format);
    // if base format and tensor size is not match, we should reflush them
    if ((isBaseFormat) && (baseFormat == ACL_FORMAT_NCDHW)) {
        // scence1: Dimensionality reduction: NCDHW->NCHW, for example: max/min
        // scence2: view, as_strided
        // NOTE(NPU Dimensionality reduction)
        if (dim == 4) {
            return ACL_FORMAT_NCHW;
        } else if (dim == 5) {
            return ACL_FORMAT_NCDHW;
        } else {
            return ACL_FORMAT_ND;
        }
    } else if (format == ACL_FORMAT_NCHW && dim != 4) {
        return ACL_FORMAT_ND;
    } else if ((dim == 0) || ((dim == 1) && (size[0] == 1) && (baseFormat == ACL_FORMAT_ND))) {
        // operators treat tensor with dimensions of 0 or shape = [1] as scalar,
        // so these tensor will stay ND format except NCHW tensor whose origin shape
        // can be expand into four dimensions.
        return ACL_FORMAT_ND;
    }
    return format;
}

FormatShape InferFormat::GuessStorageSizeWhenConvertFormat(const at::Tensor &tensor)
{
    auto format = FormatHelper::GetFormat(tensor);
    auto size = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.base_sizes_;
    auto dtype = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.data_type_;
    // TransData: ND->NZ, ND size < 2, we can expand dimension to 2, the storage have no effect.
    // now, only ND->NZ and NZ->ND will call transdata， so we no need to check other format.
    if ((size.size() < 2) && format == ACL_FORMAT_ND) {
        do {
            size.emplace_back(1);
        } while (size.size() < 2);
    }
    return FormatHelper::GetStorageSizes(format, size, dtype);
}

bool InferFormat::IsDefiniteTensorWhenMetaDataChanges(const at::Tensor &tensor, const c10::IntArrayRef &size)
{
    auto baseformat = FormatHelper::GetBaseFormat(tensor);
    if (baseformat == ACL_FORMAT_NCHW && size.size() >= 5) {
        return true;
    }
    if (baseformat == ACL_FORMAT_NCDHW && size.size() != 5) {
        return true;
    }
    return false;
}

} // namespace native
} // namespace at_npu
