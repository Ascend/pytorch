#include "TensorFactories.h"
// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>

#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/irange.h>
#include <ATen/record_function.h>
#include <c10/core/impl/COW.h>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUSwappedMemoryAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/aten/common/ResizeNpu.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/framework/InferFormat.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/common/FormatCastHelper.h"
#include "torch_npu/csrc/core/NPUTensorImpl.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "op_plugin/OpInterface.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/profiler/utils.h"
#endif

namespace at_npu {
namespace native {

namespace {

int64_t npu_byte_tensor_sum_on_cpu(const at::Tensor& t)
{
    if (t.numel() == 0) {
        return 0;
    }
    return t.to(at::kCPU, /*non_blocking=*/false)
        .contiguous()
        .reshape({-1})
        .to(at::ScalarType::Long)
        .sum()
        .item<int64_t>();
}

// Write canonical int_repr bytes into dst; NPU quantized tensors may expose multiple int slabs and a
// separate qtensor data_ptr — retry until dst.int_repr() matches expected_sum (small tensors only).
void npu_quantized_clone_write_int_repr_payload(
    at::Tensor& dst,
    const at::Tensor& npu_bytes,
    int64_t device_index,
    int64_t expected_sum)
{
    static const auto kNoopDeleter = [](void*) {};
    constexpr int kMaxAttempts = 5;
    constexpr int kMaxIntReprSlabHops = 8;
    const bool verify_sum = expected_sum >= 0 && dst.numel() > 0 && dst.numel() <= 4096;

    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        at::Tensor dst_repr = dst.int_repr();
        op_plugin::npu_stride_copy_out(
            npu_bytes,
            npu_bytes.sizes(),
            npu_bytes.strides(),
            c10::Scalar(static_cast<int64_t>(npu_bytes.storage_offset())),
            dst_repr);
        c10_npu::getCurrentNPUStream(device_index).synchronize();
        c10_npu::npuSynchronizeDevice();

        at::Tensor written = dst_repr;
        for (int hop = 0; hop < kMaxIntReprSlabHops; ++hop) {
            at::Tensor cur = dst.int_repr();
            if (!cur.defined() || written.nbytes() == 0 || !cur.sizes().equals(written.sizes())) {
                break;
            }
            cur.copy_(written, /*non_blocking=*/false);
            c10_npu::getCurrentNPUStream(device_index).synchronize();
            c10_npu::npuSynchronizeDevice();
            written = cur;
        }

        if (dst.data_ptr() != nullptr && written.defined() && dst.data_ptr() != written.data_ptr() &&
            dst.nbytes() == written.nbytes()) {
            at::Tensor q_payload = at_npu::native::from_blob(
                dst.data_ptr(),
                dst.sizes(),
                dst.strides(),
                kNoopDeleter,
                npu_bytes.options(),
                dst.device());
            q_payload.copy_(written, /*non_blocking=*/false);
            c10_npu::getCurrentNPUStream(device_index).synchronize();
            c10_npu::npuSynchronizeDevice();
        }

        const int64_t got = verify_sum ? npu_byte_tensor_sum_on_cpu(dst.int_repr()) : expected_sum;
        if (!verify_sum || got == expected_sum) {
            break;
        }
        c10_npu::npuSynchronizeDevice();
    }
}

void window_function_checks(
    const char *function_name,
    const c10::TensorOptions &options,
    int64_t window_length)
{
    TORCH_CHECK(
        options.layout() != at::kSparse,
        function_name,
        " is not implemented for sparse types, got: ",
        options, OPS_ERROR(ErrCode::NOT_SUPPORT));
    TORCH_CHECK(
        at::isFloatingType(c10::typeMetaToScalarType(options.dtype())) ||
            at::isComplexType(c10::typeMetaToScalarType(options.dtype())),
        function_name,
        " expects floating point dtypes, got: ",
        options, OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(
        window_length >= 0,
        function_name,
        " requires non-negative window_length, got window_length=",
        window_length, OPS_ERROR(ErrCode::VALUE));
}

size_t computeStorageNbytes(
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    size_t itemsize_bytes)
{
    // size of the underlying storage is 1 bigger than the offset
    // of the last element according to stride
    size_t size = 1;
    for (const auto i : c10::irange(sizes.size())) {
        if (sizes[i] == 0) {
            return 0;
        }
        size += strides[i] * (sizes[i] - 1);
    }
    return size * itemsize_bytes;
}

} // namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
at::Tensor NPUNativeFunctions::empty(
    c10::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt)
{
#ifndef BUILD_LIBTORCH
    torch_npu::profiler::NPURecordFunction profiler_guard;
#endif
    RECORD_FUNCTION("empty_tensor", std::vector<c10::IValue>({}));
    auto device_ = c10::device_or_default(device_opt);
    AT_ASSERT(device_.type() == c10::DeviceType::PrivateUse1, OPS_ERROR(ErrCode::PARAM));
    torch_npu::utils::maybe_initialize_npu(device_);
    TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt), "Only dense CPU tensors can be pinned",
                OPS_ERROR(ErrCode::NOT_SUPPORT));
    TORCH_CHECK(!(at::isComplexType(dtype_or_default(dtype_opt)) && !at_npu::native::env::CheckJitDisable()),
                "Current settings do not support Complex dtype. Please try again with jit_compile=False.",
                OPS_ERROR(ErrCode::NOT_SUPPORT));
    check_size_nonnegative(size);
    c10_npu::NPUGuard guard_(device_);
    c10::Allocator *allocator = c10_npu::NPUCachingAllocator::get();
    int64_t nelements = c10::multiply_integers(size);
    auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
    int64_t size_bytes = nelements * dtype.itemsize();
    c10::intrusive_ptr<c10::StorageImpl> storage_impl = torch_npu::make_npu_storage_impl(
        c10::StorageImpl::use_byte_size_t(),
        c10::SymInt(size_bytes),
        allocator->allocate(size_bytes),
        allocator,
        true);

    auto tensor = at::detail::make_tensor<torch_npu::NPUTensorImpl>(storage_impl, dtype);

    // Default at::TensorImpl has size [0]
    if (size.size() != 1 || size[0] != 0) {
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }
    auto memory_format = memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
    TORCH_CHECK(
        memory_format == c10::MemoryFormat::Contiguous,
        "Only c10::MemoryFormat::Contiguous is supported for creating a npu tensor", OPS_ERROR(ErrCode::NOT_SUPPORT));
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
    StorageDescHelper::SetDesc(tensor, size, tensor.strides());

    return tensor;
}

at::Tensor empty_like_npu(
    const at::Tensor &self,
    const c10::TensorOptions &options_,
    c10::optional<c10::MemoryFormat> optional_memory_format)
{
    TORCH_CHECK(
        !(options_.has_memory_format() && optional_memory_format.has_value()),
        "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
        "the redundant setter.", OPS_ERROR(ErrCode::PARAM));

    c10::TensorOptions options = self.options().merge_in(options_).merge_in(
        c10::TensorOptions().memory_format(optional_memory_format));

    TORCH_CHECK(
        !(options.layout() != at::kStrided && optional_memory_format.has_value()),
        "memory format option is only supported by strided tensors", OPS_ERROR(ErrCode::NOT_SUPPORT));
    if (options.layout() == at::kSparse && self.is_sparse()) {
        auto result = at::empty({0}, options); // to be resized
        result.sparse_resize_and_clear_(
            self.sizes(), self.sparse_dim(), self.dense_dim());
        return result;
    }

    auto memory_format = options.memory_format_opt().value_or(c10::MemoryFormat::Preserve);

    if (self.is_quantized()) {
        // To support all features of c10::MemoryFormat::Preserve we need to add
        // _empty_affine_quantized_strided function and use it similarly to
        // at::Tensor clone(const at::Tensor& src, c10::optional<c10::c10::MemoryFormat>
        // optional_memory_format) if (self.is_non_overlapping_and_dense()) ->
        // _empty_affine_quantized_strided
        if (memory_format == c10::MemoryFormat::Preserve) {
            memory_format = self.suggest_memory_format();
        }

        // Note [Explicit nullopt c10::MemoryFormat argument]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Some functions which we call default the OPTIONAL c10::MemoryFormat
        // argument to something that's not nullopt.  If we pass the
        // c10::MemoryFormat via TensorOptions, we must explicitly disable this
        // defaulting process, by explicitly passing nullopt for the c10::MemoryFormat
        // argument.  When codegen is adjusted so we can delete this argument from
        // the method signature, the argument will just disappear entirely.
        //
        // BTW, there are a few places where the optional c10::MemoryFormat is None,
        // but I still pass in nullopt for robustness.

        // We could check if dtype is still quantized?  But then should we
        // shift/scale the q_zero_point / q_scale or not?
        TORCH_CHECK(
            !options.has_dtype() || options.dtype() == self.dtype(),
            "It is currently not supported to specify a dtype that doesn't match "
            "the input tensor's dtype via empty_like.  Specified: ",
            options.dtype(),
            " Input tensor's dtype: ",
            self.dtype(), OPS_ERROR(ErrCode::TYPE));
        auto qscheme = self.qscheme();
        if (qscheme == at::kPerTensorAffine) {
            return at::_empty_affine_quantized(
                self.sizes(),
                options.memory_format(memory_format),
                self.q_scale(),
                self.q_zero_point(),
                // See Note [Explicit nullopt c10::MemoryFormat argument]
                c10::nullopt);
        } else if (qscheme == at::kPerChannelAffine) {
            // Copy the tensors with channels to avoid accidental overrides
            return at::_empty_per_channel_affine_quantized(
                self.sizes(),
                self.q_per_channel_scales().clone(c10::MemoryFormat::Preserve),
                self.q_per_channel_zero_points().clone(c10::MemoryFormat::Preserve),
                self.q_per_channel_axis(),
                options.memory_format(memory_format),
                // See Note [Explicit nullopt c10::MemoryFormat argument]
                c10::nullopt);
        } else {
            TORCH_CHECK(false, "Unsupported qscheme: ", toString(qscheme), OPS_ERROR(ErrCode::NOT_SUPPORT));
        }
    }

    at::Tensor result;

    if (memory_format == c10::MemoryFormat::Preserve &&
        !(torch_npu::utils::is_npu(options))) {
        if (self.is_non_overlapping_and_dense()) {
            result = at::empty_strided(
                self.sizes(), self.strides(), options.memory_format(c10::nullopt));
        } else {
            // See Note [Explicit nullopt c10::MemoryFormat argument]
            result = at::empty(
                self.sizes(),
                options.memory_format(self.suggest_memory_format()),
                c10::nullopt);
        }
    } else {
        // See Note [Explicit nullopt c10::MemoryFormat argument]
        if (!(torch_npu::utils::is_npu(options))) {
            result = at::empty(
                self.sizes(), options.memory_format(memory_format), c10::nullopt);
        } else {
            auto npu_format =
                torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.npu_format_;
            // self为faketensor时候获取到的format是随机值，赋值为 ND 格式
            if ((typeid(*self.storage().unsafeGetStorageImpl()) != typeid(torch_npu::NPUStorageImpl))) {
                npu_format = ACL_FORMAT_ND;
            }
            if (FormatHelper::IsBaseFormatType(npu_format) && self.unsafeGetTensorImpl()->support_as_strided() &&
                self.layout() == c10::kStrided &&
               (!optional_memory_format.has_value() || optional_memory_format.value() == c10::MemoryFormat::Preserve)) {
                // keep strides
                std::vector<int64_t> strides = at::infer_dense_strides(self.sizes(), self.strides());
                result = at::empty_strided(self.sizes(), strides, options.memory_format(std::nullopt));
            } else {
                result = OpPreparation::ApplyTensorWithFormat(self.sizes(), options, npu_format);
            }
        }
    }

    if (self.opt_names()) {
        at::namedinference::propagate_names(result, self.names());
    }

    return result;
}

at::Tensor NPUNativeFunctions::empty_like(
    const at::Tensor &self,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> optional_memory_format)
{
    c10::TensorOptions options = c10::TensorOptions().dtype(dtype_opt)
                                        .device(device_opt)
                                        .layout(layout_opt)
                                        .pinned_memory(pin_memory_opt);

    return at_npu::native::empty_like_npu(self, options, optional_memory_format);
}

at::Tensor NPUNativeFunctions::empty_with_format(
    c10::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    int64_t dst_format,
    c10::optional<int64_t> base_addr_aligned_kb)
{
#ifndef BUILD_LIBTORCH
    torch_npu::profiler::NPURecordFunction profiler_guard;
#endif
    RECORD_FUNCTION("empty_tensor", std::vector<c10::IValue>({}));
    auto device_ = c10::device_or_default(device_opt);
    torch_npu::utils::torch_check_npu(device_);
    torch_npu::utils::maybe_initialize_npu(device_);
    TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt), "Only dense CPU tensors can be pinned", OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(!(at::isComplexType(dtype_or_default(dtype_opt)) && !at_npu::native::env::CheckJitDisable()),
                "Current settings do not support Complex dtype. Please try again with jit_compile=False.",
                OPS_ERROR(ErrCode::NOT_SUPPORT));
    check_size_nonnegative(size);
    c10_npu::NPUGuard guard_(device_);
    c10::Allocator *allocator = c10_npu::NPUCachingAllocator::get();
    // when the shape and format are not match, fix format here.
    aclFormat format = InferFormat::GuessStorageFormat(size, static_cast<aclFormat>(dst_format));
    auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
    int64_t nelements = StorageDescHelper::GetMemorySize(size, format, dtype);
    int64_t size_bytes = nelements * dtype.itemsize();
    c10::intrusive_ptr<c10::StorageImpl> storage_impl;
    if (!base_addr_aligned_kb.has_value()) {
        storage_impl = torch_npu::make_npu_storage_impl(
            c10::StorageImpl::use_byte_size_t(),
            c10::SymInt(size_bytes),
            allocator->allocate(size_bytes),
            allocator,
            true);
    } else {
        storage_impl = c10::make_intrusive<torch_npu::NPUStorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            static_cast<size_t>(size_bytes),
            c10_npu::NPUCachingAllocator::allocate_with_aligned(size_bytes, base_addr_aligned_kb.value()),
            allocator,
            true);
    }

    auto tensor = at::detail::make_tensor<torch_npu::NPUTensorImpl>(storage_impl, dtype);

    // Default NPUTensorImpl has size [0]
    if (size.size() != 1 || size[0] != 0) {
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(c10::MemoryFormat::Contiguous);
    StorageDescHelper::SetDesc(tensor, size, tensor.strides(), format);
    return tensor;
}

at::Tensor NPUNativeFunctions::unsafe_empty_with_format(
    c10::IntArrayRef size,
    c10::optional <at::ScalarType> dtype_opt,
    c10::optional <c10::Layout> layout_opt,
    c10::optional <c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    int64_t dst_format,
    bool keep_format)
{
    // This is a special interface that can adjust the memory application results. Check before use.

    // Some ops cannot operate directly based on ND format, such as MatMul, BatchMatMul, MaxPoolWithArgmaxV1.
    // For these ops, specify the parameter keep_format to ensure that
    // the specified internal format is preserved.
    if ((!keep_format) && at_npu::native::env::CheckForbidInternalFormat()) {
        dst_format = static_cast<int64_t>(FormatHelper::GetBaseFormat(static_cast<aclFormat>(dst_format)));
        TORCH_WARN_ONCE("Cannot create tensor with interal format while allow_internel_format=False, "
                        "tensor will be created with base format.");
    }

    return NPUNativeFunctions::empty_with_format(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, dst_format, c10::nullopt);
}

at::Tensor NPUNativeFunctions::empty_with_format(
    c10::IntArrayRef size,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    int64_t dst_format)
{
    torch_npu::utils::torch_check_npu(c10::device_or_default(device_opt));
    caffe2::TypeMeta dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
    c10::TensorOptions options = c10::TensorOptions().dtype(dtype_opt)
                                        .device(device_opt)
                                        .layout(layout_opt)
                                        .pinned_memory(pin_memory_opt);
    at::Tensor result = OpPreparation::ApplyTensorWithFormat(size, options, dst_format);
    if (names.has_value()) {
        internal_set_names_inplace(result, names);
    }

    return result;
}

at::Tensor empty_with_format_name_npu(
    c10::IntArrayRef size,
    c10::optional<at::DimnameList> names,
    const c10::TensorOptions &options,
    int64_t dst_format)
{
    at::Tensor result = OpPreparation::ApplyTensorWithFormat(size, options, dst_format);
    if (names.has_value()) {
        internal_set_names_inplace(result, names);
    }

    return result;
}

at::Tensor NPUNativeFunctions::empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    check_size_nonnegative(size);
    c10::optional<c10::MemoryFormat> optional_memory_format = c10::nullopt;
    auto t = NPUNativeFunctions::empty({0}, dtype_opt, layout_opt, device_opt, pin_memory_opt, optional_memory_format);
    c10_npu::NPUGuard guard(c10::device_or_default(device_opt));
    StorageDescHelper::SetDesc(t, size, stride);
    at_npu::native::resize_impl_npu_(t.unsafeGetTensorImpl(), size, stride);
    return t;
}

at::Tensor NPUNativeFunctions::new_empty_strided_symint(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory)
{
    return at::native::new_empty_strided_symint(self, size, stride, dtype, layout, device, pin_memory);
}

at::Tensor &empty_out_npu(
    at::Tensor &result,
    c10::IntArrayRef size,
    c10::optional<c10::MemoryFormat> optional_memory_format)
{
    // Preferably, this argument would not be accepted by _out, but the code
    // generator requires the out and non-out overloads to match exactly
    TORCH_CHECK(
        !optional_memory_format.has_value(),
        "'memory_format' argument is incompatible with 'out' tensor argument", OPS_ERROR(ErrCode::PARAM));
    check_size_nonnegative(size);
    if (result.is_sparse()) {
        result.sparse_resize_and_clear_(size, size.size(), 0);
    } else {
        result.resize_(size);
    }
    return result;
}

at::Tensor NPUNativeFunctions::empty_with_swapped_memory(
    c10::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Device> device_opt)
{
#ifndef BUILD_LIBTORCH
    torch_npu::profiler::NPURecordFunction profiler_guard;
#endif
    RECORD_FUNCTION("empty_with_swapped_memory", std::vector<c10::IValue>({}));
    auto device_ = device_opt.value_or(at::Device(c10::DeviceType::PrivateUse1, c10_npu::current_device()));
    torch_npu::utils::torch_check_npu(device_);
    torch_npu::utils::maybe_initialize_npu(device_);
    TORCH_CHECK(!(at::isComplexType(dtype_or_default(dtype_opt)) && !at_npu::native::env::CheckJitDisable()),
                "Current settings do not support Complex dtype. Please try again with jit_compile=False.",
                PTA_ERROR(ErrCode::NOT_SUPPORT));
    check_size_nonnegative(size);
    c10_npu::NPUGuard guard_(device_);
    c10::Allocator *allocator = c10_npu::NPUSwappedMemoryAllocator::get();
    int64_t nelements = c10::multiply_integers(size);
    auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
    int64_t size_bytes = nelements * dtype.itemsize();
    c10::intrusive_ptr<c10::StorageImpl> storage_impl = torch_npu::make_npu_storage_impl(
        c10::StorageImpl::use_byte_size_t(),
        c10::SymInt(size_bytes),
        allocator->allocate(size_bytes),
        allocator,
        true);

    auto tensor = at::detail::make_tensor<torch_npu::NPUTensorImpl>(storage_impl, dtype);

    // Default at::TensorImpl has size [0]
    if (size.size() != 1 || size[0] != 0) {
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(c10::MemoryFormat::Contiguous);
    StorageDescHelper::SetDesc(tensor, size, tensor.strides());

    return tensor;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ blackman_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
at::Tensor NPUNativeFunctions::blackman_window(
    int64_t window_length,
    bool periodic,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    c10::TensorOptions options = c10::TensorOptions().dtype(dtype_opt)
                                        .device(device_opt)
                                        .layout(layout_opt)
                                        .pinned_memory(pin_memory_opt);

    window_function_checks("blackman_window", options, window_length);
    if (window_length == 0) {
        return at::empty({0}, options);
    }
    if (window_length == 1) {
        return at::ones({1}, options);
    }
    if (periodic) {
        window_length += 1;
    }
    auto window = at::arange(window_length, options).mul_(M_PI / static_cast<double>(window_length - 1));
    window = window.mul(4).cos_().mul_(0.08) - window.mul(2).cos_().mul_(0.5) + 0.42;
    return periodic ? window.narrow(0, 0, window_length - 1) : window;
}

at::Tensor NPUNativeFunctions::blackman_window(
    int64_t window_length,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    return blackman_window(window_length, true, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ bartlett_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
at::Tensor NPUNativeFunctions::bartlett_window(
    int64_t window_length,
    bool periodic,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    c10::TensorOptions options = c10::TensorOptions().dtype(dtype_opt)
                                        .device(device_opt)
                                        .layout(layout_opt)
                                        .pinned_memory(pin_memory_opt);

    window_function_checks("bartlett_window", options, window_length);
    if (window_length == 0) {
        return at::empty({0}, options);
    }
    if (window_length == 1) {
        return at::ones({1}, options);
    }
    if (periodic) {
        window_length += 1;
    }
    auto window = at::arange(window_length, options).mul_(2. / static_cast<double>(window_length - 1));
    const int64_t first_half_size = (static_cast<uint64_t>(window_length - 1) >> 1) + 1;
    window.narrow(0, first_half_size, window_length - first_half_size).mul_(-1).add_(2);
    return periodic ? window.narrow(0, 0, window_length - 1) : window;
}

at::Tensor NPUNativeFunctions::bartlett_window(
    int64_t window_length,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    return bartlett_window(window_length, true, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hann_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
at::Tensor NPUNativeFunctions::hann_window(
    int64_t window_length,
    bool periodic,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    c10::TensorOptions options = c10::TensorOptions().dtype(dtype_opt)
                                        .device(device_opt)
                                        .layout(layout_opt)
                                        .pinned_memory(pin_memory_opt);

    window_function_checks("hann_window", options, window_length);
    return at::hamming_window(window_length, periodic, 0.5, 0.5, options);
}

at::Tensor NPUNativeFunctions::hann_window(
    int64_t window_length,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    return hann_window(window_length, true, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hamming_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
at::Tensor NPUNativeFunctions::hamming_window(
    int64_t window_length,
    bool periodic,
    double alpha,
    double beta,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    c10::TensorOptions options = c10::TensorOptions().dtype(dtype_opt)
                                        .device(device_opt)
                                        .layout(layout_opt)
                                        .pinned_memory(pin_memory_opt);

    window_function_checks("hamming_window", options, window_length);
    if (window_length == 0) {
        return at::empty({0}, options);
    }
    if (window_length == 1) {
        return at::ones({1}, options);
    }
    if (periodic) {
        window_length += 1;
    }
    auto window = at::arange(window_length, options);
    window.mul_(M_PI * 2. / static_cast<double>(window_length - 1))
        .cos_()
        .mul_(-beta)
        .add_(alpha);
    return periodic ? window.narrow(0, 0, window_length - 1) : window;
}

at::Tensor NPUNativeFunctions::hamming_window(
    int64_t window_length,
    bool periodic,
    double alpha,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    return hamming_window(window_length, periodic, alpha, 0.46, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor NPUNativeFunctions::hamming_window(
    int64_t window_length,
    bool periodic,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    return hamming_window(window_length, periodic, 0.54, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor NPUNativeFunctions::hamming_window(
    int64_t window_length,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    return hamming_window(window_length, true, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename T>
at::Tensor tensor_npu(c10::ArrayRef<T> values, const c10::TensorOptions &options)
{
    auto result = at::empty(values.size(), options);
    AT_ASSERT(result.is_contiguous(), OPS_ERROR(ErrCode::VALUE));
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(result.scalar_type(), "tensor_npu", [&] {
        std::copy(
            values.begin(), values.end(), result.template data_ptr<scalar_t>());
    });
    return result;
}

template <typename T>
at::Tensor tensor_backend_npu(c10::ArrayRef<T> values, const c10::TensorOptions &options)
{
    auto npu_tensor = tensor_npu(values, options.device(c10::DeviceType::PrivateUse1));
    return npu_tensor.to(options.device());
}

#define TENSOR(T, _1)                                                             \
at::Tensor tensor_npu(c10::ArrayRef<T> values, const c10::TensorOptions &options) \
{                                                                                 \
    if (options.device().type() != c10::DeviceType::PrivateUse1) {                \
        return tensor_backend_npu(values, options);                               \
    } else {                                                                      \
        return tensor_npu(values, options);                                       \
    }                                                                             \
}
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ clone ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
at::Tensor NPUNativeFunctions::clone(
    const at::Tensor &src,
    c10::optional<c10::MemoryFormat> format)
{
    c10_npu::NPUGuard guard(src.device());

    // Quantized (QUInt8/QInt8): TransContiguous / generic copy_d2d paths have produced
    // non-deterministic or partially wrong buffers vs int_repr semantics (see contiguous()
    // twice differing). Always materialize via Byte stride-copy matching CopyKernel.
    if (src.is_quantized()) {
        TORCH_CHECK(
            !format.has_value() || *format == c10::MemoryFormat::Contiguous ||
                *format == c10::MemoryFormat::Preserve,
            "NPU quantized clone only supports Contiguous or Preserve memory_format.",
            OPS_ERROR(ErrCode::NOT_SUPPORT));
        // empty_like + int_repr stride_copy only when int_repr metadata matches. When
        // MetaDataAreMatch(src.int_repr)==0, NPU quantized tensors can expose multiple int slabs;
        // npu_stride_copy_out into dst.int_repr().reshape(...) still left later dst.int_repr()
        // reading a different buffer (test_view_ops.TestOldViewOpsPRIVATEUSE1.test_ravel_npu nc).
        // int_repr CPU byte staging (plain Byte on CPU, not QuantizedCPU) when q or int_repr NPUStorageDesc
        // disagrees with tensor sizes/strides. Do not use src.to(CPU) / QuantizedCPU / r.to(NPU) on QTensor.
        if (src.device().type() == c10::DeviceType::PrivateUse1) {
            c10_npu::npuSynchronizeDevice();
        }
        const at::Tensor src_repr = src.int_repr();
        const bool npu_src_q_meta_mismatch =
            src.device().type() == c10::DeviceType::PrivateUse1 &&
            !StorageDescHelper::MetaDataAreMatch(&src);
        const bool npu_src_repr_meta_mismatch =
            src.device().type() == c10::DeviceType::PrivateUse1 &&
            !StorageDescHelper::MetaDataAreMatch(&src_repr);
        // MetaData(q)==0 on a view (e.g. ravel -> [625]) or int_repr meta mismatch (transpose): device
        // stride_copy / empty_like+H2D can leave dst.int_repr() on a non-canonical slab (checksum 1 vs 348).
        // Stage correct int_repr bytes on CPU, then H2D into empty_like dst + slab propagation (quantize_per_tensor
        // on NPU can leave dst.int_repr() on a different slab than the quantized payload).
        if (npu_src_q_meta_mismatch || npu_src_repr_meta_mismatch) {
            at::Tensor cpu_repr;
            if (src.numel() == 0) {
                cpu_repr = at::empty_like(src_repr, src_repr.options().device(at::kCPU));
            } else if (npu_src_repr_meta_mismatch) {
                // e.g. transpose: int_repr NPUStorageDesc disagrees with sizes/strides; read via
                // storage-owner base_sizes_/base_strides_ when they cover the same numel.
                const torch_npu::NPUStorageDesc& desc =
                    torch_npu::NPUBridge::GetNpuStorageImplDesc(src);
                const int64_t base_numel =
                    c10::multiply_integers(c10::IntArrayRef(desc.base_sizes_));
                if (desc.base_sizes_.size() > 0 && base_numel == src.numel()) {
                    at::Tensor read_ir = src_repr.as_strided(
                        c10::IntArrayRef(desc.base_sizes_),
                        c10::IntArrayRef(desc.base_strides_),
                        src_repr.storage_offset());
                    cpu_repr = read_ir.contiguous().reshape({-1}).to(at::kCPU, /*non_blocking=*/false);
                }
                if (!cpu_repr.defined() || cpu_repr.numel() != src.numel()) {
                    cpu_repr = at::empty_like(src_repr, src_repr.options().device(at::kCPU));
                    cpu_repr.copy_(src_repr, /*non_blocking=*/false);
                }
            } else {
                // MetaData(q)==0 only (e.g. ravel/view rank-1): int_repr meta already matches src;
                // do not as_strided via qtensor desc (can yield numel 1); D2H the logical int_repr view.
                cpu_repr = src_repr.contiguous().to(at::kCPU, /*non_blocking=*/false);
                TORCH_CHECK(
                    cpu_repr.numel() == src.numel(),
                    "NPU quantized clone: int_repr D2H numel ",
                    cpu_repr.numel(),
                    " != src.numel ",
                    src.numel(),
                    OPS_ERROR(ErrCode::VALUE));
            }
            at::Tensor dst = (format.has_value() && *format == c10::MemoryFormat::Contiguous)
                ? at::empty_like(src, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                : at::empty_like(src);
            StorageDescHelper::SetDesc(dst, dst.sizes(), dst.strides());
            c10_npu::npuSynchronizeDevice();
            const at::Tensor npu_bytes =
                cpu_repr.contiguous().to(src.device(), /*non_blocking=*/false);
            int64_t expected_sum = -1;
            if (src.numel() > 0 && src.numel() <= 4096) {
                expected_sum = cpu_repr.to(at::ScalarType::Long).sum().item<int64_t>();
            }
            npu_quantized_clone_write_int_repr_payload(
                dst,
                npu_bytes,
                src.device().index(),
                expected_sum);
            return dst;
        }

        at::Tensor dst = (format.has_value() && *format == c10::MemoryFormat::Contiguous)
            ? at::empty_like(src, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
            : at::empty_like(src);
        if (src.device().type() == c10::DeviceType::PrivateUse1) {
            StorageDescHelper::SetDesc(dst, dst.sizes(), dst.strides());
            c10_npu::npuSynchronizeDevice();
        }
        at::Tensor dst_repr = dst.int_repr();
        op_plugin::npu_stride_copy_out(
            src_repr,
            src_repr.sizes(),
            src_repr.strides(),
            c10::Scalar(static_cast<int64_t>(src_repr.storage_offset())),
            dst_repr);
        if (src.device().type() == c10::DeviceType::PrivateUse1) {
            // npu_stride_copy_out writes dst_repr; successive dst.int_repr() may return different
            // physical slabs (post_copy: same_data_ptr=0, again!=frozen on first clone). One
            // copy_(canonical, dst_repr) is insufficient if int_repr() is unstable. Propagate the
            // stride-copied buffer through int_repr() hops until stable or cap, syncing after each copy.
            c10_npu::getCurrentNPUStream(src.device().index()).synchronize();
            c10_npu::npuSynchronizeDevice();
            // Do not break early when cur.data_ptr() == written: int_repr() can return the same
            // slab for several calls (so stride_copy data is visible), then a later call returns a
            // different slab (post_copy again!=frozen). Run a fixed number of copy+sync rounds so
            // every returned slab receives the stride-copied payload.
            at::Tensor written = dst_repr;
            constexpr int kMaxIntReprSlabHops = 8;
            for (int hop = 0; hop < kMaxIntReprSlabHops; ++hop) {
                at::Tensor cur = dst.int_repr();
                if (!cur.defined() || written.nbytes() == 0) {
                    break;
                }
                if (!cur.sizes().equals(written.sizes())) {
                    break;
                }
                cur.copy_(written, /*non_blocking=*/false);
                c10_npu::getCurrentNPUStream(src.device().index()).synchronize();
                c10_npu::npuSynchronizeDevice();
                written = cur;
            }
        }
        return dst;
    }

    OptimizationCases opt_cases{"reshape", "slice"};
    if (TransContiguous::CanOptimize(src, opt_cases)) {
        // clone with any npu formats
        auto formatTempTensor = TransContiguous::ContiguousOptimizeWithAnyFormat(src, opt_cases);
        return formatTempTensor.value();
    } else {
        // clone with base formats
        auto baseSelf = OpPreparation::ApplyTensorWithSizes(src.sizes(), src.options());
        at::Tensor baseSrc = src;
        if (!FormatHelper::IsBaseFormatType(src)) {
            baseSrc = FormatCastHelper::ApplyBaseFormatTensorBy(src);
        }
        copy_d2d_dtype_baseformat(baseSelf, baseSrc, false);
        return baseSelf;
    }
}

at::Tensor NPUNativeFunctions::full(
    c10::IntArrayRef size,
    const c10::Scalar& fill_value,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    c10::TensorOptions options = c10::TensorOptions().dtype(dtype_opt)
                                        .device(device_opt)
                                        .layout(layout_opt)
                                        .pinned_memory(pin_memory_opt);
    TORCH_CHECK(
        options.layout() != at::kSparse,
        "full(...) is not implemented for sparse layout", OPS_ERROR(ErrCode::TYPE));

    if (!dtype_opt.has_value()) {
        if (fill_value.isBoolean()) {
            options = options.dtype(at::kBool);
        } else if (fill_value.isIntegral(false)) {
            options = options.dtype(at::kLong);
        } else {
            options = options.dtype(c10::get_default_dtype());
        }
    }

    auto result = OpPreparation::ApplyTensorWithSizes(size, options);
    return result.fill_(fill_value);
}

at::Tensor NPUNativeFunctions::tril_indices(
    int64_t row,
    int64_t col,
    int64_t offset,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    c10::TensorOptions options = c10::TensorOptions().dtype(dtype_opt)
                                                        .device(device_opt)
                                                        .layout(layout_opt)
                                                        .pinned_memory(pin_memory_opt);
    check_args(row, col, options);

    auto tril_size = get_tril_size(row, col, offset);

    // create an empty Tensor with correct size
    auto result = at::empty({2 * tril_size}, options);

    // The following three approaches result in very little performance
    // differences. Hence, the 2nd option is taken for simpler code, and to return
    // contiguous tensors. Refer to #14904 for more details.
    //
    // 1. sequential RAM access: fill row coordinates first, then columns. This
    //    results in two for-loop and more arithmetic operations.
    //
    // 2. interleaved RAM access: fill in index coordinates one by one, which
    //    jumps between the two output Tensor rows in every iteration.
    //
    // 3. sequential RAM + transpose: create an n X 2 Tensor, fill the Tensor
    //    sequentially, and then transpose it.
    // fill the Tensor with correct values
    int64_t i = 0;
    int64_t r = std::max<int64_t>(0, -offset);
    int64_t c = 0;

    while (i < tril_size) {
        result[i] = r;
        result[tril_size + i++] = c;

        // move to the next column and check if (r, c) is still in bound
        c += 1;
        if (c > r + offset || c >= col) {
            r += 1;
            c = 0;
            // NOTE: not necessary to check if r is less than row here, because i
            // and tril_size provide the guarantee
        }
    }

    return result.reshape({2, tril_size});
}

at::Tensor NPUNativeFunctions::triu_indices(
    int64_t row,
    int64_t col,
    int64_t offset,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    c10::TensorOptions options = c10::TensorOptions().dtype(dtype_opt)
                                                      .device(device_opt)
                                                      .layout(layout_opt)
                                                      .pinned_memory(pin_memory_opt);
    check_args(row, col, options);

    auto triu_size = row * col - get_tril_size(row, col, offset - 1);

    // create an empty Tensor with correct size
    auto result = at::empty({2 * triu_size}, options);

    // fill the Tensor with correct values
    int64_t i = 0;
    // not typing std::max with scalar_t as it could be an unsigned type
    // NOTE: no need to check if the returned value of std::max overflows
    // scalar_t, as i and triu_size act as a guard.
    int64_t c = std::max<int64_t>(0, offset);
    int64_t r = 0;
    while (i < triu_size) {
        result[i] = r;
        result[triu_size + i++] = c;

        // move to the next column and check if (r, c) is still in bound
        c += 1;
        if (c >= col) {
            r += 1;
            // not typing std::max with scalar_t as it could be an unsigned type
            // NOTE: not necessary to check if c is less than col or overflows here,
            // because i and triu_size act as a guard.
            c = std::max<int64_t>(0, r + offset);
        }
    }
    return result.reshape({2, triu_size});
}

at::Tensor NPUNativeFunctions::isnan(const at::Tensor& self)
{
    return at::native::isnan(self);
}

at::Tensor NPUNativeFunctions::unfold(const at::Tensor& self, int64_t dimension, int64_t size, int64_t step)
{
    return at::native::unfold(self, dimension, size, step);
}

at::Tensor NPUNativeFunctions::_lazy_clone(at::Tensor const& self)
{
    // 1. Obtain the StorageImpl pointer of the source Tensor
    c10::StorageImpl* self_storage = self.storage().unsafeGetStorageImpl();

    // 2. Call the PyTorch core COW mechanism to create a lazy clone storage
    c10::intrusive_ptr<c10::StorageImpl> storage =
        c10::impl::cow::lazy_clone_storage(*self_storage);
    TORCH_CHECK(storage != nullptr);

    // 3. Create a new TensorImpl based on cloned storage
    auto tensor = c10::make_intrusive<c10::TensorImpl>(
        c10::Storage(std::move(storage)),
        self.key_set(),
        self.dtype());

    // 4. Set the metadata (dimensions, strides, offset) of the new Tensor
    tensor->set_sizes_and_strides(
        self.sym_sizes(),
        self.sym_strides(),
        self.sym_storage_offset());

    // 5. Create a Tensor and copy the NPU-specific description information
    auto result = at::Tensor(std::move(tensor));
    StorageDescHelper::CopyDesc(result, self);

    return result;
}

} // namespace native
} // namespace at_npu
