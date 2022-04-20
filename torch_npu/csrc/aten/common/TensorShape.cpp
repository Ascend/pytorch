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

#include "third_party/acl/inc/acl/acl_base.h"

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
  if (InferFormat::IsDefiniteTensorWhenMetaDataChanges(dst, size)) {
    dst = FormatCastHelper::ApplyBaseFormatTensorBy(dst);
  }
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
  at::native::setStrided(result, size, stride, storage_offset);
  return result;
}

at::Tensor& NPUNativeFunctions::as_strided_(
    at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<int64_t> storage_offset_) {
  if (InferFormat::IsDefiniteTensorWhenMetaDataChanges(self, size)) {
    self = FormatCastHelper::CovertSelfToBaseFormat(self);
  }
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  at::native::setStrided(self, size, stride, storage_offset);
  return self;
}

} // namespace native
} // namespace at_npu