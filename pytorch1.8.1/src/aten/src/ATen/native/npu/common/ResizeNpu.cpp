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
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/npu/common/ResizeNpu.h>
#include "ATen/native/npu/frame/FormatHelper.h"
#include <torch/script.h>

namespace at {
namespace native {

using namespace at::native::npu;

Tensor& resize_npu_(
    Tensor& self,
    IntArrayRef size,
    c10::optional<c10::MemoryFormat> format) {
  // because of resize _impl_npu_ only support at base format, so
  // no need to reflush NpuStorageDesc here.
  if (!FormatHelper::IsBaseFormatType(self)) {
    self.npu_format_cast_(FormatHelper::GetBaseFormat(self));
  }
  auto* self_ = self.unsafeGetTensorImpl();
  resize_impl_npu_(self_, size, /*strides=*/c10::nullopt);
  // self_->maybe_zero_dim(size.size() == 0);
  return self;
}

Tensor& resize_as_npu_(
    Tensor& self,
    const Tensor& the_template,
    c10::optional<c10::MemoryFormat> format) {
  TORCH_CHECK(
      !(self.is_sparse() || the_template.is_sparse()),
      "NPU does not support sparse tensors.");
  TORCH_CHECK(
      !format.has_value(), "NPU does not support specify memory_format.");

  Tensor& result = self.resize_(the_template.sizes());
  at::namedinference::propagate_names(result, the_template);
  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("resize_", TORCH_FN(resize_npu_));
}

} // namespace native
} // namespace at
