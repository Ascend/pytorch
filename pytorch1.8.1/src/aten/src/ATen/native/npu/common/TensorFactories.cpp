// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "TensorFactories.h"
// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/Exception.h>
#include <c10/npu/NPUCachingAllocator.h>
#include <ATen/native/npu/common/ResizeNpu.h>
#include <ATen/native/npu/frame/StorageDescHelper.h>
#include <ATen/native/npu/frame/InferFormat.h>
#include <ATen/native/npu/common/InnerNpuNativeFunction.h>
#include <ATen/record_function.h>
#include "ATen/native/npu/utils/OpAdapter.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>

namespace at {
namespace native {
using namespace at::native::npu;
namespace {
void window_function_checks(
    const char* function_name,
    const TensorOptions& options,
    int64_t window_length) {
  TORCH_CHECK(
      options.layout() != kSparse,
      function_name,
      " is not implemented for sparse types, got: ",
      options);
  TORCH_CHECK(
      at::isFloatingType(typeMetaToScalarType(options.dtype())) ||
          at::isComplexType(typeMetaToScalarType(options.dtype())),
      function_name,
      " expects floating point dtypes, got: ",
      options);
  TORCH_CHECK(
      window_length >= 0,
      function_name,
      " requires non-negative window_length, got window_length=",
      window_length);
}

} // namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor empty_npu(IntArrayRef size,
                 c10::optional<ScalarType> dtype_opt,
                 c10::optional<Layout> layout_opt,
                 c10::optional<Device> device_opt,
                 c10::optional<bool> pin_memory_opt,
                 c10::optional<c10::MemoryFormat> memory_format_opt) {
  AT_ASSERT(device_or_default(device_opt).type() == DeviceType::NPU);
  TORCH_CHECK(!pinned_memory_or_default(pin_memory_opt), "Only dense CPU tensors can be pinned");
  check_size_nonnegative(size);
  c10::Allocator* allocator = at::npu::NPUCachingAllocator::get();
  int64_t nelements = prod_intlist(size);
  auto dtype = scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      true);
  auto tensor =
      detail::make_tensor<TensorImpl>(storage_impl, DispatchKey::NPUTensorId, dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }
  auto memory_format =
      memory_format_opt.value_or(MemoryFormat::Contiguous);
  TORCH_CHECK(
      memory_format == MemoryFormat::Contiguous,
      "Only MemoryFormat::Contiguous is supported for creating a npu tensor");
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  StorageDescHelper::SetDesc(tensor, size, tensor.strides());

  return tensor;
}

Tensor empty_like_npu(
    const Tensor& self,
    const TensorOptions& options_,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");

  TensorOptions options = self.options().merge_in(options_).merge_in(
      TensorOptions().memory_format(optional_memory_format));

  TORCH_CHECK(
      !(options.layout() != kStrided && optional_memory_format.has_value()),
      "memory format option is only supported by strided tensors");
  if (options.layout() == kSparse && self.is_sparse()) {
    auto result = at::empty({0}, options); // to be resized
    result.sparse_resize_and_clear_(
        self.sizes(), self.sparse_dim(), self.dense_dim());
    return result;
  }

  auto memory_format =
      options.memory_format_opt().value_or(MemoryFormat::Contiguous);

  if (self.is_quantized()) {
    // TODO: To support all features of MemoryFormat::Preserve we need to add
    // _empty_affine_quantized_strided function and use it similarly to
    // Tensor clone(const Tensor& src, c10::optional<c10::MemoryFormat>
    // optional_memory_format) if (self.is_non_overlapping_and_dense()) ->
    // _empty_affine_quantized_strided
    if (memory_format == MemoryFormat::Preserve) {
      memory_format = self.suggest_memory_format();
    }

    // Note [Explicit nullopt MemoryFormat argument]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Some functions which we call default the OPTIONAL MemoryFormat
    // argument to something that's not nullopt.  If we pass the
    // MemoryFormat via TensorOptions, we must explicitly disable this
    // defaulting process, by explicitly passing nullopt for the MemoryFormat
    // argument.  When codegen is adjusted so we can delete this argument from
    // the method signature, the argument will just disappear entirely.
    //
    // BTW, there are a few places where the optional MemoryFormat is None,
    // but I still pass in nullopt for robustness.

    // We could check if dtype is still quantized?  But then should we
    // shift/scale the q_zero_point / q_scale or not?
    TORCH_CHECK(
        !options.has_dtype() || options.dtype() == self.dtype(),
        "It is currently not supported to specify a dtype that doesn't match "
        "the input tensor's dtype via empty_like.  Specified: ",
        options.dtype(),
        " Input tensor's dtype: ",
        self.dtype());
    auto qscheme = self.qscheme();
    if (qscheme == kPerTensorAffine) {
      return at::_empty_affine_quantized(
          self.sizes(),
          options.memory_format(memory_format),
          self.q_scale(),
          self.q_zero_point(),
          // See Note [Explicit nullopt MemoryFormat argument]
          c10::nullopt);
    } else if (qscheme == kPerChannelAffine) {
      // Copy the tensors with channels to avoid accidental overrides
      return at::_empty_per_channel_affine_quantized(
          self.sizes(),
          self.q_per_channel_scales().clone(at::MemoryFormat::Preserve),
          self.q_per_channel_zero_points().clone(at::MemoryFormat::Preserve),
          self.q_per_channel_axis(),
          options.memory_format(memory_format),
          // See Note [Explicit nullopt MemoryFormat argument]
          c10::nullopt);
    } else {
      TORCH_CHECK(false, "Unsupported qscheme: ", toString(qscheme));
    }
  }

  Tensor result;

  if (memory_format == MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      result = at::empty_strided(
          self.sizes(), self.strides(), options.memory_format(c10::nullopt));
    } else {
      // See Note [Explicit nullopt MemoryFormat argument]
      result = at::empty(
          self.sizes(),
          options.memory_format(self.suggest_memory_format()),
          c10::nullopt);
    }
  } else {
    // See Note [Explicit nullopt MemoryFormat argument]
    if (!options.device().is_npu()) {
      result = at::empty(
          self.sizes(), options.memory_format(memory_format), c10::nullopt);
    } else {
      auto npu_format =
          self.storage().unsafeGetStorageImpl()->npu_desc_.npu_format_;
      result = at::empty_with_format(self.sizes(), self.options(), npu_format);
    }
  }

  if (self.opt_names()) {
    namedinference::propagate_names(result, self.names());
  }

  return result;
}

Tensor empty_like_new_npu(
    const Tensor& self,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> optional_memory_format) {

  TensorOptions options;
  options.dtype(dtype_opt);
  auto device = device_or_default(device_opt);
  options.device(device);
  options.layout(layout_opt);
  options.pinned_memory(pin_memory_opt);

  return at::native::empty_like_npu(self, options, optional_memory_format);
}

Tensor empty_new_with_format_npu(IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    int64_t dst_format) {
  AT_ASSERT(device_or_default(device_opt).type() == DeviceType::NPU);
  TORCH_CHECK(!pinned_memory_or_default(pin_memory_opt), "Only dense CPU tensors can be pinned");
  check_size_nonnegative(size);
  c10::Allocator* allocator = at::npu::NPUCachingAllocator::get();
  // when the shape and format are not match, fix format here.
  aclFormat format = InferFormat::GuessStorageFormat(size, (aclFormat)dst_format);
  int64_t nelements = StorageDescHelper::GetMemorySize(size, format);
  auto dtype = scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      true);
  auto tensor =
      detail::make_tensor<TensorImpl>(storage_impl, DispatchKey::NPUTensorId, dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }
  auto memory_format = MemoryFormat::Contiguous;
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  StorageDescHelper::SetDesc(tensor, size, tensor.strides(), format);

  return tensor;
}

Tensor empty_with_format_npu(IntArrayRef size,
    const TensorOptions& options,
    int64_t dst_format) {
  AT_ASSERT(options.device().type() == DeviceType::NPU);
  AT_ASSERT(options.backend() == at::Backend::NPU);
  TORCH_CHECK(!options.pinned_memory(), "Only dense CPU tensors can be pinned");
  check_size_nonnegative(size);
  c10::Allocator* allocator = at::npu::NPUCachingAllocator::get();
  // when the shape and format are not match, fix format here.
  aclFormat format = InferFormat::GuessStorageFormat(size, (aclFormat)dst_format);
  int64_t nelements = StorageDescHelper::GetMemorySize(size, format);
  auto dtype = options.dtype();
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      true);
  auto tensor =
      detail::make_tensor<TensorImpl>(storage_impl, DispatchKey::NPUTensorId, dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }
  auto memory_format = MemoryFormat::Contiguous;
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  StorageDescHelper::SetDesc(tensor, size, tensor.strides(), format);

  return tensor;
}

Tensor empty_new_with_format_name_npu(IntArrayRef size,
    c10::optional<DimnameList> names,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    int64_t dst_format) {
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
  TensorOptions options;
  options.dtype(dtype);
  auto device = device_or_default(device_opt);
  options.device(device);
  options.layout(layout_opt);
  options.pinned_memory(pin_memory_opt);
  Tensor result =
      at::empty_with_format(size, options, dst_format);
  if (names.has_value()) {
    internal_set_names_inplace(result, names);
  }

  return result;
}
Tensor empty_with_format_npu(IntArrayRef size,
    c10::optional<DimnameList> names,
    const TensorOptions& options,
    int64_t dst_format) {
  Tensor result =
      at::empty_with_format(size, options, dst_format);
  if (names.has_value()) {
    internal_set_names_inplace(result, names);
  }

  return result;
}

Tensor empty_with_format_name_npu(IntArrayRef size,
    c10::optional<DimnameList> names,
    const TensorOptions& options,
    int64_t dst_format) {
  Tensor result =
      at::empty_with_format(size, options, dst_format);
  if (names.has_value()) {
    internal_set_names_inplace(result, names);
  }

  return result;
}

Tensor empty_strided_npu(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  check_size_nonnegative(size);
  c10::optional<c10::MemoryFormat> optional_memory_format = c10::nullopt;
  auto t = at::native::empty_npu({0},
                        dtype_opt,
                        layout_opt,
                        device_opt,
                        pin_memory_opt,
						optional_memory_format);
  StorageDescHelper::SetDesc(t, size, stride);
  at::native::resize_impl_npu_(t.unsafeGetTensorImpl(), size, stride);
  return t;
}

Tensor& empty_out_npu(
    Tensor& result,
    IntArrayRef size,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  // Preferably, this argument would not be accepted by _out, but the code
  // generator requires the out and non-out overloads to match exactly
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "'memory_format' argument is incompatible with 'out' tensor argument");
  check_size_nonnegative(size);
  if (result.is_sparse()) {
    result.sparse_resize_and_clear_(size, size.size(), 0);
  } else {
    result.resize_(size);
  }
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ blackman_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor blackman_window_periodic_npu(int64_t window_length,
    bool periodic,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  TensorOptions options;
  options.dtype(dtype_opt);
  auto device = device_or_default(device_opt);
  options.device(device);
  options.layout(layout_opt);
  options.pinned_memory(pin_memory_opt);

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

Tensor blackman_window_npu(int64_t window_length,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  return blackman_window_periodic_npu(window_length, /*periodic=*/true, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ bartlett_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor bartlett_window_periodic_npu(
    int64_t window_length,
    bool periodic,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  TensorOptions options;
  options.dtype(dtype_opt);
  auto device = device_or_default(device_opt);
  options.device(device);
  options.layout(layout_opt);
  options.pinned_memory(pin_memory_opt);

  window_function_checks("bartlett_window", options, window_length);
  if (window_length == 0) {
    return at::empty({0}, options);
  }
  if (window_length == 1) {
    return native::ones({1}, options);
  }
  if (periodic) {
    window_length += 1;
  }
  auto window = at::arange(window_length, options).mul_(2. / static_cast<double>(window_length - 1));
  const int64_t first_half_size = ((unsigned int64_t)(window_length - 1) >> 1) + 1;
  window.narrow(0, first_half_size, window_length - first_half_size).mul_(-1).add_(2);
  return periodic ? window.narrow(0, 0, window_length - 1) : window;
}

Tensor bartlett_window_npu(int64_t window_length,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  return bartlett_window_periodic_npu(window_length, /*periodic=*/true, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hann_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor hann_window_periodic_npu(
    int64_t window_length,
    bool periodic,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  TensorOptions options;
  options.dtype(dtype_opt);
  auto device = device_or_default(device_opt);
  options.device(device);
  options.layout(layout_opt);
  options.pinned_memory(pin_memory_opt);

  window_function_checks("hann_window", options, window_length);
  return at::hamming_window(window_length, periodic, 0.5, 0.5, options);
}

Tensor hann_window_npu(int64_t window_length,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  return hann_window_periodic_npu(window_length, true, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hamming_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor hamming_window_beta_npu(
    int64_t window_length,
    bool periodic,
    double alpha,
    double beta,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  TensorOptions options;
  options.dtype(dtype_opt);
  auto device = device_or_default(device_opt);
  options.device(device);
  options.layout(layout_opt);
  options.pinned_memory(pin_memory_opt);

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


Tensor hamming_window_alpha_npu(
    int64_t window_length,
    bool periodic,
    double alpha,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  return hamming_window_beta_npu(window_length, periodic, alpha, /*beta=*/0.46, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

Tensor hamming_window_periodic_npu(
    int64_t window_length,
    bool periodic,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  return hamming_window_alpha_npu(window_length, periodic, /*alpha=*/0.54, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}



Tensor hamming_window_npu(int64_t window_length,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  return hamming_window_periodic_npu(window_length, /*periodic=*/true, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename T>
Tensor tensor_npu(ArrayRef<T> values, const TensorOptions& options) {
  auto result = at::empty(values.size(), options);
  AT_ASSERT(result.is_contiguous());
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(result.scalar_type(), "tensor_npu", [&] {
    std::copy(
        values.begin(), values.end(), result.template data_ptr<scalar_t>());
  });
  return result;
}

template <typename T>
Tensor tensor_backend_npu(ArrayRef<T> values, const TensorOptions& options) {
  auto npu_tensor = tensor_npu(values, options.device(DeviceType::NPU));
  return npu_tensor.to(options.device());
}

#define TENSOR(T, _1)                                                   \
  Tensor tensor_npu(ArrayRef<T> values, const TensorOptions& options) { \
    if (options.device().type() != c10::DeviceType::NPU) {              \
      return tensor_backend_npu(values, options);                       \
    } else {                                                            \
      return tensor_npu(values, options);                               \
    }                                                                   \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ clone ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor clone_npu(const Tensor& src, c10::optional<c10::MemoryFormat> format) {
  auto desc = src.storage().unsafeGetStorageImpl()->npu_desc_;
  auto formatSelf = OpPreparation::ApplyTensorWithFormat(
      src.sizes(), src.options(), desc.npu_format_);
  if (try_to_optimize_copy_with_any_format(formatSelf, src)) {
    return formatSelf;
  } else if (can_use_memcpy(formatSelf, src)) {
    RECORD_FUNCTION("d2dCopyAsync with format", std::vector<c10::IValue>({src}));
    copy_d2d_by_memcpy(formatSelf, src);
    return formatSelf;
  } else {
    auto baseSelf = OpPreparation::ApplyTensor(src);
    copy_d2d_dtype(baseSelf, src, false);
    return baseSelf;
  }
}

Tensor full_npu(
    IntArrayRef size,
    Scalar fill_value,
    const TensorOptions& options) {
  TORCH_CHECK(
      options.layout() != kSparse,
      "full(...) is not implemented for sparse layout");

  auto result = OpPreparation::ApplyTensorWithSizes(size, options);
  return result.fill_(fill_value);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("bartlett_window", TORCH_FN(bartlett_window_npu));
  m.impl("bartlett_window.periodic", TORCH_FN(bartlett_window_periodic_npu));
  m.impl("blackman_window", TORCH_FN(blackman_window_npu));
  m.impl("blackman_window.periodic", TORCH_FN(blackman_window_periodic_npu));
  m.impl("hamming_window", TORCH_FN(hamming_window_npu));
  m.impl("hamming_window.periodic", TORCH_FN(hamming_window_periodic_npu));
  m.impl("hamming_window.periodic_alpha", TORCH_FN(hamming_window_alpha_npu));
  m.impl("hamming_window.periodic_alpha_beta", TORCH_FN(hamming_window_beta_npu));
  m.impl("hann_window", TORCH_FN(hann_window_npu));
  m.impl("hann_window.periodic", TORCH_FN(hann_window_periodic_npu));
  m.impl("empty.memory_format", TORCH_FN(empty_npu));
  m.impl("clone", TORCH_FN(clone_npu));
  m.impl("empty_strided", TORCH_FN(empty_strided_npu));
  m.impl("empty_with_format", TORCH_FN(empty_new_with_format_npu));
  m.impl("empty_with_format.names", TORCH_FN(empty_new_with_format_name_npu));
  m.impl("empty_like", TORCH_FN(empty_like_new_npu));
}
} // namespace native
} // namespace at
