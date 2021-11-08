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
#include <torch/csrc/autograd/record_function.h>
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
Tensor empty_npu(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  AT_ASSERT(options.device().type() == DeviceType::NPU);
  AT_ASSERT(options.backend() == at::Backend::NPU);
  // AT_ASSERT(!options.is_variable());  // is_variable should have been
  // 'unpacked'  // TODO: remove this when Variable and Tensor are merged
  TORCH_CHECK(!options.pinned_memory(), "Only dense CPU tensors can be pinned");
  check_size_nonnegative(size);

  c10::Allocator* allocator = at::npu::NPUCachingAllocator::get();
  int64_t nelements = prod_intlist(size);
  auto dtype = options.dtype();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      dtype,
      nelements,
      allocator->allocate(nelements * dtype.itemsize()),
      allocator,
      /*resizeable=*/true);

  auto tensor =
      detail::make_tensor<TensorImpl>(storage_impl, DispatchKey::NPUTensorId);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Contiguous);
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

Tensor empty_with_format_npu(
    IntArrayRef size,
    const TensorOptions& options,
    int64_t dst_format) {
  AT_ASSERT(options.device().type() == DeviceType::NPU);
  AT_ASSERT(options.backend() == at::Backend::NPU);
  // AT_ASSERT(!options.is_variable());  // is_variable should have been
  // 'unpacked'  // TODO: remove this when Variable and Tensor are merged
  TORCH_CHECK(!options.pinned_memory(), "Only dense CPU tensors can be pinned");
  check_size_nonnegative(size);
  c10::Allocator* allocator = at::npu::NPUCachingAllocator::get();
  // when the shape and format are not match, fix format here.
  aclFormat format = InferFormat::GuessStorageFormat(size, (aclFormat)dst_format);
  int64_t nelements = StorageDescHelper::GetMemorySize(size, format);
  auto dtype = options.dtype();
  const auto& storage_impl = c10::make_intrusive<StorageImpl>(
      dtype,
      nelements,
      allocator->allocate(nelements * dtype.itemsize()),
      allocator,
      /*resizeable=*/true);

  auto tensor =
      detail::make_tensor<TensorImpl>(storage_impl, DispatchKey::NPUTensorId);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  tensor.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::Contiguous);
  StorageDescHelper::SetDesc(tensor, size, tensor.strides(), format);
  return tensor;
}

Tensor empty_with_format_npu(
    IntArrayRef size,
    optional<DimnameList> names,
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
    const TensorOptions& options) {
  check_size_nonnegative(size);
  auto t = at::native::empty_npu({0}, options);
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

Tensor blackman_window_npu(int64_t window_length, const TensorOptions& options) {
  return blackman_window_npu(window_length, /*periodic=*/true, options);
}

Tensor blackman_window_npu(int64_t window_length, bool periodic, const TensorOptions& options) {
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ bartlett_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor bartlett_window_npu(int64_t window_length, const TensorOptions& options) {
  return bartlett_window_npu(window_length, /*periodic=*/true, options);
}

Tensor bartlett_window_npu(
    int64_t window_length,
    bool periodic,
    const TensorOptions& options) {
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hann_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor hann_window_npu(int64_t window_length, const TensorOptions& options) {
  return hann_window_npu(window_length, true, options);
}

Tensor hann_window_npu(
    int64_t window_length,
    bool periodic,
    const TensorOptions& options) {
  window_function_checks("hann_window", options, window_length);
  return at::hamming_window(window_length, periodic, 0.5, 0.5, options);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hamming_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor hamming_window_npu(int64_t window_length, const TensorOptions& options) {
  return hamming_window_npu(window_length, /*periodic=*/true, options);
}

Tensor hamming_window_npu(
    int64_t window_length,
    bool periodic,
    const TensorOptions& options) {
  return hamming_window_npu(window_length, periodic, /*alpha=*/0.54, options);
}

Tensor hamming_window_npu(
    int64_t window_length,
    bool periodic,
    double alpha,
    const TensorOptions& options) {
  return hamming_window_npu(window_length, periodic, alpha, /*beta=*/0.46, options);
}

Tensor hamming_window_npu(
    int64_t window_length,
    bool periodic,
    double alpha,
    double beta,
    const TensorOptions& options) {
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor tril_indices_npu(
    int64_t row, int64_t col, int64_t offset, const TensorOptions& options) {
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
  int64_t r = std::max<int64_t>(0, -offset), c = 0;

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

Tensor triu_indices_npu(
    int64_t row, int64_t col, int64_t offset, const TensorOptions& options) {
  check_args(row, col, options);

  auto triu_size = row * col - get_tril_size(row, col, offset - 1);

  // create an empty Tensor with correct size
  auto result = at::empty({2 * triu_size}, options);

  // fill the Tensor with correct values
  int64_t i = 0;
  // not typing std::max with scalar_t as it could be an unsigned type
  // NOTE: no need to check if the returned value of std::max overflows
  // scalar_t, as i and triu_size act as a guard.
  int64_t c = std::max<int64_t>(0, offset), r = 0;
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
    RECORD_HOST_FUNCTION("d2dCopyAsync with format", std::vector<c10::IValue>({src}));
    copy_d2d_by_memcpy(formatSelf, src);
    return formatSelf;
  } else {
    auto baseSelf = OpPreparation::ApplyTensorWithSizes(src.sizes(), src.options());
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

  auto result = at::empty_with_format(size, options);
  return result.fill_(fill_value);
}

} // namespace native
} // namespace at
