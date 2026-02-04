#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Resize.h>
#include <ATen/quantized/QTensorImpl.h>
#include <c10/util/Optional.h>
#include <algorithm>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/aten/common/FormatCastHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/common/ResizeNpu.h"
#include "third_party/acl/inc/acl/acl_base.h"

namespace {
// Named type instead of a pair/tuple so that we can be sure to
// construct the vectors in place and get NRVO.
struct InferUnsqueezeGeometryResult {
    at::DimVector sizes;
    at::DimVector strides;
    InferUnsqueezeGeometryResult(c10::IntArrayRef tensor_sizes, c10::IntArrayRef tensor_strides)
        : sizes(tensor_sizes.begin(), tensor_sizes.end()), strides(tensor_strides.begin(), tensor_strides.end()) {}
};
}

InferUnsqueezeGeometryResult inferUnsqueezeGeometry(const at::Tensor& tensor, int64_t dim)
{
    InferUnsqueezeGeometryResult result(tensor.sizes(), tensor.strides());
    int64_t new_stride = dim >= tensor.dim() ? 1 : result.sizes[dim] * result.strides[dim];
    result.sizes.insert(result.sizes.begin() + dim, 1);
    result.strides.insert(result.strides.begin() + dim, new_stride);

    return result;
}

std::tuple<at::DimVector, at::DimVector> inferSqueezeGeometry(const at::Tensor &tensor)
{
    at::DimVector sizes;
    at::DimVector strides;

    for (const auto d : c10::irange(tensor.dim())) {
        if (tensor.sizes()[d] != 1) {
            sizes.push_back(tensor.sizes()[d]);
            strides.push_back(tensor.strides()[d]);
        }
    }

    return std::make_tuple(std::move(sizes), std::move(strides));
}

std::tuple<at::DimVector, at::DimVector> inferSqueezeGeometry(const at::Tensor& tensor, int64_t dim)
{
    at::DimVector sizes;
    at::DimVector strides;

    for (const auto d : c10::irange(tensor.dim())) {
        if (d != dim || tensor.sizes()[dim] != 1) {
            sizes.push_back(tensor.sizes()[d]);
            strides.push_back(tensor.strides()[d]);
        }
    }
    return std::make_tuple(std::move(sizes), std::move(strides));
}

namespace at_npu {
namespace native {

at::Tensor alias_with_sizes_and_strides_npu(
    const at::Tensor& self,
    const c10::IntArrayRef sizes,
    const c10::IntArrayRef strides)
{
    at::Tensor self_;
    if (self.is_quantized()) {
        self_ = at::detail::make_tensor<at::QTensorImpl>(
            c10::TensorImpl::VIEW,
            c10::Storage(self.storage()),
            self.key_set(),
            self.dtype(),
            get_qtensorimpl(self)->quantizer());
        auto* self_tmp_ = self_.unsafeGetTensorImpl();
        self_tmp_->set_storage_offset(self.storage_offset());
        self_tmp_->set_sizes_and_strides(sizes, strides);
    } else {
        self_ = at::detail::make_tensor<at::TensorImpl>(
            c10::TensorImpl::VIEW,
            c10::Storage(self.storage()),
            self.key_set(),
            self.dtype());
        auto* self_tmp_ = self_.unsafeGetTensorImpl();
        self_tmp_->set_storage_offset(self.storage_offset());
        self_tmp_->set_sizes_and_strides(sizes, strides);
    }
    at::namedinference::propagate_names(self_, self);
    return self_;
}

at::Tensor NPUNativeFunctions::view(const at::Tensor& self, c10::IntArrayRef size)
{
    auto inferred_size = at::infer_size(size, self.numel());
    auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
    TORCH_CHECK(
        stride.has_value(),
        "view size is "
        "not compatible with input tensor's size and stride (at least one dimension"
        " spans across two contiguous subspaces). Use .reshape(...) instead.", OPS_ERROR(ErrCode::PARAM));
    auto stride_value = *stride;
    auto dst = self;
    return alias_with_sizes_and_strides_npu(dst, inferred_size, stride_value);
}

at::Tensor NPUNativeFunctions::as_strided(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<int64_t> storage_offset_)
{
    auto dst = self;
    if (InferFormat::IsDefiniteTensorWhenMetaDataChanges(dst, size) && !FormatHelper::IsOpInputBaseFormat(dst)) {
        TORCH_WARN_ONCE("current tensor is running as_strided, don't perform inplace operations on the returned value."
            " If you encounter this warning and have precision issues,"
            " you can try torch.npu.config.allow_internal_format = False to resolve precision issues.")
        dst = FormatCastHelper::ApplyBaseFormatTensorBy(dst);
    }
    auto storage_offset = storage_offset_.value_or(dst.storage_offset());
    auto result = at::detail::make_tensor<at::TensorImpl>(
        c10::TensorImpl::VIEW,
        c10::Storage(dst.storage()),
        dst.key_set(),
        dst.dtype());
    setStrided(result, size, stride, storage_offset);
    return result;
}

const at::Tensor& NPUNativeFunctions::as_strided__symint(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset_)
{
    auto ks = self.key_set();
    bool is_fake_or_meta = ks.has_all(c10::DispatchKeySet(c10::BackendComponent::MetaBit)) ||
                           ks.has_all(c10::DispatchKeySet(c10::DispatchKey::Python)) ||
                           self.is_meta();
    if (!is_fake_or_meta) {
        if (InferFormat::IsDefiniteTensorWhenMetaDataChanges(self, c10::asIntArrayRefUnchecked(size)) &&
            !FormatHelper::IsOpInputBaseFormat(self)) {
            TORCH_CHECK(false, "Current tensor is running as_strided__symint while internal format is not allowed."
                " You can try torch.npu.config.allow_internal_format = False to avoid the problem.",
                PTA_ERROR(ErrCode::NOT_SUPPORT));
        }
    }
    auto storage_offset = storage_offset_.value_or(self.sym_storage_offset());
    at::native::setStrided(self, size, stride, std::move(storage_offset));
    return self;
}

at::Tensor NPUNativeFunctions::unsqueeze(const at::Tensor& self, int64_t dim)
{
    dim = at::maybe_wrap_dim(dim, self.dim() + 1);
    auto g = inferUnsqueezeGeometry(self, dim);
    return self.as_strided(g.sizes, g.strides);
}

at::Tensor NPUNativeFunctions::squeeze(const at::Tensor& self)
{
    auto g = inferSqueezeGeometry(self);
    at::Tensor result = self.as_strided(std::get<0>(g), std::get<1>(g));
    auto maybe_outnames = at::namedinference::compute_squeeze_outnames(self);
    at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
    return result;
}

at::Tensor NPUNativeFunctions::squeeze(const at::Tensor& self, int64_t dim)
{
    int64_t dims = self.dim();
    dim = at::maybe_wrap_dim(dim, dims);
    if (dims == 0 || self.sizes()[dim] != 1) {
        return self.as_strided(self.sizes(), self.strides());
    }
    auto g = inferSqueezeGeometry(self, dim);
    auto result = self.as_strided(std::get<0>(g), std::get<1>(g));
    at::namedinference::propagate_names_except(result, self, {dim});
    return result;
}

at::Tensor NPUNativeFunctions::_reshape_alias(const at::Tensor& self, at::IntArrayRef sizes, at::IntArrayRef strides)
{
    return self.view(sizes);
}

} // namespace native
} // namespace at_npu
