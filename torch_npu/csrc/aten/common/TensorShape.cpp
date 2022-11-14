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

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Resize.h>
#include <ATen/quantized/QTensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <algorithm>
#include <vector>

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
      : sizes(tensor_sizes.begin(), tensor_sizes.end())
      , strides(tensor_strides.begin(), tensor_strides.end()) {}
};
}

InferUnsqueezeGeometryResult inferUnsqueezeGeometry(const at::Tensor& tensor, int64_t dim) {
  InferUnsqueezeGeometryResult result(tensor.sizes(), tensor.strides());
  int64_t new_stride = dim >= tensor.dim() ? 1 : result.sizes[dim] * result.strides[dim];
  result.sizes.insert(result.sizes.begin() + dim, 1);
  result.strides.insert(result.strides.begin() + dim, new_stride);

  return result;
}

std::tuple<at::DimVector, at::DimVector> inferSqueezeGeometry(const at::Tensor &tensor) {
  at::DimVector sizes;
  at::DimVector strides;

  for(const auto d : c10::irange(tensor.dim())) {
    if(tensor.sizes()[d] != 1) {
      sizes.push_back(tensor.sizes()[d]);
      strides.push_back(tensor.strides()[d]);
    }
  }

  return std::make_tuple(std::move(sizes), std::move(strides));
}

std::tuple<at::DimVector, at::DimVector> inferSqueezeGeometry(const at::Tensor& tensor, int64_t dim) {
  at::DimVector sizes;
  at::DimVector strides;

  for(const auto d : c10::irange(tensor.dim())) {
    if(d != dim || tensor.sizes()[dim] != 1) {
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
    const c10::IntArrayRef strides) {
  at::Tensor self_;
  if (self.is_quantized()) {
    auto impl = c10::make_intrusive<at::QTensorImpl>(
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype(),
        get_qtensorimpl(self)->quantizer());
    impl->set_storage_offset(self.storage_offset());
    impl->set_sizes_and_strides(sizes, strides);
    self_ = at::Tensor(std::move(impl));
  } else {
    auto impl = c10::make_intrusive<at::TensorImpl>(
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype());
    impl->set_storage_offset(self.storage_offset());
    impl->set_sizes_and_strides(sizes, strides);
    self_ = at::Tensor(std::move(impl));
  }
  at::namedinference::propagate_names(self_, self);
  return self_;
}

at::Tensor NPUNativeFunctions::view(const at::Tensor& self, c10::IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto stride_value = *stride;
  auto dst = self;
  return alias_with_sizes_and_strides_npu(dst, inferred_size, stride_value);
}

at::Tensor NPUNativeFunctions::as_strided(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<int64_t> storage_offset_) {
  auto dst = self;
  if (InferFormat::IsDefiniteTensorWhenMetaDataChanges(dst, size)) {
    dst = FormatCastHelper::ApplyBaseFormatTensorBy(dst);
  }
  auto storage_offset = storage_offset_.value_or(dst.storage_offset());
  auto result = at::detail::make_tensor<at::TensorImpl>(
      c10::Storage(dst.storage()),
      dst.key_set(),
      dst.dtype());
  setStrided(result, size, stride, storage_offset);
  return result;
}

const at::Tensor& NPUNativeFunctions::as_strided_(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<int64_t> storage_offset_) {
  at::Tensor result = self;
  if (InferFormat::IsDefiniteTensorWhenMetaDataChanges(result, size)) {
    result = FormatCastHelper::CovertSelfToBaseFormat(result);
  }
  auto storage_offset = storage_offset_.value_or(result.storage_offset());
  at::native::setStrided(result, size, stride, storage_offset);
  return result;
}

at::Tensor NPUNativeFunctions::unsqueeze(const at::Tensor& self, int64_t dim) {
    dim = at::maybe_wrap_dim(dim, self.dim() + 1);
    auto g = inferUnsqueezeGeometry(self, dim);
    return self.as_strided(g.sizes, g.strides);

}

at::Tensor NPUNativeFunctions::squeeze(const at::Tensor& self) {
  auto g = inferSqueezeGeometry(self);
  at::Tensor result = self.as_strided(std::get<0>(g), std::get<1>(g));
  auto maybe_outnames = at::namedinference::compute_squeeze_outnames(self);
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

at::Tensor NPUNativeFunctions::squeeze(const at::Tensor& self, int64_t dim) {
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

} // namespace native
} // namespace at_npu