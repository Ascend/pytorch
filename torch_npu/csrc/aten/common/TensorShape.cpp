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
#include <ATen/quantized/Quantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/aten/common/FormatCastHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/common/ResizeNpu.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "op_plugin/OpInterface.h"

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
    if (!FormatHelper::IsOpInputBaseFormat(dst)) {
        if (InferFormat::IsDefiniteTensorWhenMetaDataChanges(dst, size)) {
            TORCH_WARN_ONCE("current tensor is running as_strided, don't perform inplace operations on the returned value."
                " If you encounter this warning and have precision issues,"
                " you can try torch.npu.config.allow_internal_format = False to resolve precision issues.")
            dst = FormatCastHelper::ApplyBaseFormatTensorBy(dst);
        }
    }
    auto storage_offset = storage_offset_.value_or(dst.storage_offset());
    if (dst.is_quantized()) {
        auto result = at::detail::make_tensor<at::QTensorImpl>(
            c10::TensorImpl::VIEW,
            c10::Storage(dst.storage()),
            dst.key_set(),
            dst.dtype(),
            get_qtensorimpl(dst)->quantizer());
        at::native::setStrided(result, size, stride, storage_offset);
        return result;
    }
    auto result = at::detail::make_tensor<at::TensorImpl>(
        c10::TensorImpl::VIEW,
        c10::Storage(dst.storage()),
        dst.key_set(),
        dst.dtype());
    at::native::setStrided(result, size, stride, storage_offset);
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
        if (!FormatHelper::IsOpInputBaseFormat(self)) {
            if (InferFormat::IsDefiniteTensorWhenMetaDataChanges(self, c10::asIntArrayRefUnchecked(size))) {
                TORCH_CHECK(false, "Current tensor is running as_strided__symint while internal format is not allowed."
                    " You can try torch.npu.config.allow_internal_format = False to avoid the problem.",
                    PTA_ERROR(ErrCode::NOT_SUPPORT));
            }
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

namespace {

// Stride pattern for row-major memory (sizes with 1 are neutral for contiguity checks).
bool npu_quantized_row_major_dense(const at::Tensor& self)
{
    if (!self.dim() || self.numel() == 0) {
        return true;
    }
    int64_t expected = 1;
    for (int64_t d = static_cast<int64_t>(self.dim()) - 1; d >= 0; --d) {
        const int64_t sz = self.size(d);
        if (sz != 1) {
            if (self.stride(d) != expected) {
                return false;
            }
            expected *= sz;
        }
    }
    return true;
}

// alias_with_sizes_and_strides_npu leaves NPUStorageDesc describing the storage owner shape; a view
// with new sizes/strides can show MetaDataAreMatch(out)==0 while int_repr still matches (see logs).
// Do not SetDesc in-place on shared storage (would corrupt the base tensor). Materialize via clone
// so the flattened QTensorImpl regains consistent NPUStorageDesc; skipping clone when int_repr
// meta matches can leave MetaData(q)==0 on rank-1 views and break assertEqual between tensors that
// alias different storages (test_view_ops nc=True quantized).
at::Tensor npu_quantized_view_materialize_if_storage_desc_mismatch(const at::Tensor& r)
{
    if (r.device().type() != c10::DeviceType::PrivateUse1) {
        return r;
    }
    const bool meta_q = at_npu::native::StorageDescHelper::MetaDataAreMatch(&r);
    if (meta_q) {
        return r;
    }
    at::Tensor out = at_npu::native::NPUNativeFunctions::clone(r, c10::MemoryFormat::Contiguous);
    return out;
}

// ATen's _unsafe_view wraps view_impl and still validates strides via computeStride. For quantized
// NPU, ravel()/view(-1) after a buggy is_contiguous() short-circuit (or contiguous() passthrough)
// must match reshape(): contiguous clone then view — but only when the inferred shape truly
// collapses rank to one dim of numel (see test_view_ops.TestOldViewOpsPRIVATEUSE1.test_ravel_npu).
at::Tensor npu_quantized_view_symint(const at::Tensor& self, c10::SymIntArrayRef size)
{
    const auto inferred_size = at::infer_size(c10::asIntArrayRefUnchecked(size), self.numel());
    const auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
    if (stride.has_value()) {
        // Match NPUNativeFunctions::view: build a QTensorImpl view via alias_with_sizes_and_strides_npu.
        // at::_unsafe_view_symint can diverge from that path and break quantized equality
        // (e.g. test_view_ops.TestOldViewOpsPRIVATEUSE1.test_ravel_npu transpose + ravel).
        at::Tensor r =
            at_npu::native::alias_with_sizes_and_strides_npu(self, inferred_size, c10::IntArrayRef(*stride));
        r = npu_quantized_view_materialize_if_storage_desc_mismatch(r);
        return r;
    }
    const bool flatten_to_rank1 = static_cast<int64_t>(inferred_size.size()) == 1 &&
        inferred_size[0] == static_cast<int64_t>(self.numel());
    TORCH_CHECK(
        flatten_to_rank1,
        "view size is "
        "not compatible with input tensor's size and stride (at least one dimension"
        " spans across two contiguous subspaces). Use .reshape(...) instead.",
        OPS_ERROR(ErrCode::PARAM));
    at::Tensor c = self.clone(c10::MemoryFormat::Contiguous);
    const auto stride_after_clone = at::detail::computeStride(c.sizes(), c.strides(), inferred_size);
    TORCH_CHECK(
        stride_after_clone.has_value(),
        "view size is "
        "not compatible with input tensor's size and stride (at least one dimension"
        " spans across two contiguous subspaces). Use .reshape(...) instead.",
        OPS_ERROR(ErrCode::PARAM));
    at::Tensor r = at_npu::native::alias_with_sizes_and_strides_npu(
        c, inferred_size, c10::IntArrayRef(*stride_after_clone));
    r = npu_quantized_view_materialize_if_storage_desc_mismatch(r);
    return r;
}

at::Tensor npu_quantized_as_strided_symint(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset)
{
    return at_npu::native::NPUNativeFunctions::as_strided(
        self,
        c10::asIntArrayRefUnchecked(size),
        c10::asIntArrayRefUnchecked(stride),
        storage_offset.has_value() ? c10::make_optional(storage_offset->expect_int()) : c10::nullopt);
}

at::Tensor npu_quantized_empty_memory_format_symint(
    c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt)
{
    const auto device = c10::device_or_default(device_opt);
    TORCH_CHECK(
        device.is_privateuseone(),
        "QuantizedPrivateUse1 empty.memory_format expects an NPU (PrivateUse1) device.");

    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory_opt),
        "Quantized tensors do not support pin_memory.");

    const auto layout = layout_opt.value_or(c10::Layout::Strided);
    TORCH_CHECK(
        layout == c10::Layout::Strided,
        "Quantized tensors only support strided layout, got ",
        layout);

    at::TensorOptions options =
        at::TensorOptions().dtype(dtype_opt).layout(layout_opt).device(device_opt).pinned_memory(pin_memory_opt);
    TORCH_CHECK(
        !(options.has_memory_format() && memory_format_opt.has_value()),
        "Cannot set memory_format both in TensorOptions and explicit argument; ",
        "please delete the redundant setter.");
    if (memory_format_opt.has_value()) {
        options = options.memory_format(*memory_format_opt);
    }

    TORCH_CHECK(
        options.has_dtype(),
        "Must provide dtype for quantized empty.memory_format.");

    auto qt = c10::typeMetaToScalarType(options.dtype());
    TORCH_CHECK(
        c10::isQIntType(qt),
        "empty.memory_format on QuantizedPrivateUse1 expects a quantized dtype, got ",
        qt);

    at::QuantizerPtr quantizer = at::make_unknown_quantizer(qt);
    return at::new_qtensor(c10::asIntArrayRefUnchecked(size), options, std::move(quantizer));
}

// empty_like / composite paths call empty_strided; PrivateUse1 has NPUNativeFunctions::empty_strided
// but QuantizedPrivateUse1 does not use that registration.
at::Tensor npu_quantized_empty_strided_symint(
    c10::SymIntArrayRef sym_size,
    c10::SymIntArrayRef sym_stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    const c10::IntArrayRef size = c10::asIntArrayRefUnchecked(sym_size);
    const c10::IntArrayRef stride = c10::asIntArrayRefUnchecked(sym_stride);
    const auto device = c10::device_or_default(device_opt);

    TORCH_CHECK(
        device.is_privateuseone(),
        "QuantizedPrivateUse1 empty_strided expects PrivateUse1 (NPU) device.");

    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory_opt),
        "Quantized tensors do not support pin_memory.");

    const auto layout = layout_opt.value_or(c10::Layout::Strided);
    TORCH_CHECK(
        layout == c10::Layout::Strided,
        "Quantized tensors only support strided layout, got ",
        layout);

    TORCH_CHECK(dtype_opt.has_value(), "Must provide dtype for quantized empty_strided.");
    TORCH_CHECK(
        c10::isQIntType(*dtype_opt),
        "QuantizedPrivateUse1 empty_strided expects a quantized dtype, got ",
        *dtype_opt);

    at::TensorOptions options =
        at::TensorOptions().dtype(dtype_opt).layout(layout).device(device).pinned_memory(pin_memory_opt);

    at::QuantizerPtr quantizer = at::make_unknown_quantizer(*dtype_opt);
    at::Tensor t = at::new_qtensor(size, options, std::move(quantizer));

    c10_npu::NPUGuard guard(device);
    at_npu::native::StorageDescHelper::SetDesc(t, size, stride);
    at_npu::native::resize_impl_npu_(t.unsafeGetTensorImpl(), size, stride);
    return t;
}

// ravel() trusts is_contiguous() before calling view(-1). If that flag disagrees with
// actual strides, return self unchanged and view fails — force materialization via clone().
at::Tensor npu_quantized_contiguous(const at::Tensor& self, c10::MemoryFormat memory_format)
{
    TORCH_CHECK(
        memory_format == c10::MemoryFormat::Contiguous,
        "NPU quantized contiguous supports Contiguous memory format only.", OPS_ERROR(ErrCode::NOT_SUPPORT));
    const bool short_circuit = self.is_contiguous(memory_format) && npu_quantized_row_major_dense(self);
    if (short_circuit) {
        return self;
    }
    return at_npu::native::NPUNativeFunctions::clone(self, memory_format);
}

// contiguous() -> clone(); without this registration, QuantizedPrivateUse1 uses the generic
// composite clone and never reaches NPUNativeFunctions::clone (int_repr stride_copy) in TensorFactories.cpp.
at::Tensor npu_quantized_clone(
    const at::Tensor& self,
    c10::optional<c10::MemoryFormat> memory_format)
{
    return at_npu::native::NPUNativeFunctions::clone(self, memory_format);
}

at::Tensor npu_quantized_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking)
{
    at::Tensor dst_mut = dst;

    if (self.is_quantized() && dst_mut.is_quantized()) {
        TORCH_CHECK(self.numel() == dst_mut.numel(), "QuantizedPrivateUse1 _copy_from: numel mismatch.");
        TORCH_CHECK(
            self.scalar_type() == dst_mut.scalar_type(),
            "QuantizedPrivateUse1 _copy_from: quantized dtype mismatch.");

        if (self.qscheme() == at::kPerTensorAffine) {
            at::set_quantizer_(
                dst_mut,
                at::make_per_tensor_affine_quantizer(
                    self.q_scale(), self.q_zero_point(), self.scalar_type()));
        } else if (self.qscheme() == at::kPerChannelAffine) {
            at::set_quantizer_(
                dst_mut,
                at::make_per_channel_affine_quantizer(
                    self.q_per_channel_scales(),
                    self.q_per_channel_zero_points(),
                    self.q_per_channel_axis(),
                    self.scalar_type()));
        } else {
            TORCH_CHECK(
                false,
                "QuantizedPrivateUse1 _copy_from: unsupported qscheme for NPU int_repr copy path.");
        }

        const at::Tensor src_repr = self.int_repr();
        at::Tensor dst_repr_same_shape = dst_mut.int_repr().reshape(self.sizes());
        op_plugin::npu_stride_copy_out(
            src_repr,
            src_repr.sizes(),
            src_repr.strides(),
            c10::Scalar(static_cast<int64_t>(src_repr.storage_offset())),
            dst_repr_same_shape);
        (void)non_blocking;
        return dst_mut;
    }

    at_npu::native::NPUNativeFunctions::copy_(dst_mut, self, non_blocking);
    return dst_mut;
}

// native_functions.yaml dispatches QuantizedCPU/CUDA empty_like -> empty_like_quantized.
// QuantizedPrivateUse1 is not listed, so composite empty_like runs (calls empty_symint ->
// empty.memory_format), which hits make_unknown_quantizer and breaks qscheme() during asserts.
// Mirror QuantizedCUDA by forwarding to NPUNativeFunctions::empty_like (same as empty_like_quantized).
at::Tensor npu_quantized_empty_like(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> optional_memory_format)
{
    return at_npu::native::NPUNativeFunctions::empty_like(
        self, dtype_opt, layout_opt, device_opt, pin_memory_opt, optional_memory_format);
}

// Composite ravel sometimes lowers to reshape(-1), which short-circuits via is_contiguous_or_false()
// and may view-flatten without the row-major clone path that Quantized+NPU tensors need after
// transpose. Match explicit contiguous().view(-1) (test_view_ops.TestOldViewOpsPRIVATEUSE1.test_ravel).
at::Tensor npu_quantized_ravel(const at::Tensor& self)
{
    // Dispatch through aten::contiguous (npu_quantized_contiguous), not NPUNativeFunctions::contiguous —
    // the latter only checks is_contiguous(), not row-major stride truth, and can return transpose as-is.
    at::Tensor materialized = self.contiguous(c10::MemoryFormat::Contiguous);
    return materialized.view({-1});
}

} // namespace

// Quantized NPU tensors carry DispatchKey::QuantizedPrivateUse1; register view to match native
// _unsafe_view (reshape copy path) and wire other factory/copy hooks used by functionalization.
TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
    m.impl("view", TORCH_FN(npu_quantized_view_symint));
    m.impl("as_strided", TORCH_FN(npu_quantized_as_strided_symint));
    m.impl("ravel", TORCH_FN(npu_quantized_ravel));
    m.impl("empty_like", TORCH_FN(npu_quantized_empty_like));
    m.impl("empty.memory_format", TORCH_FN(npu_quantized_empty_memory_format_symint));
    m.impl("empty_strided", TORCH_FN(npu_quantized_empty_strided_symint));
    m.impl("contiguous", TORCH_FN(npu_quantized_contiguous));
    m.impl("clone", TORCH_FN(npu_quantized_clone));
    m.impl("_copy_from", TORCH_FN(npu_quantized_copy_from));
}
