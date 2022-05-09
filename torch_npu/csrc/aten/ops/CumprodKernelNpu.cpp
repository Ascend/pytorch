// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeFunctions::_cumprod_out(const at::Tensor& self, int64_t dim, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Cumprod")
    .Input(self)
    .Output(result)
    .Attr("axis", dim)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::cumprod_out(const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor& result) {
  at::ScalarType dstType;
  if (dtype.has_value()) {
    dstType = dtype.value();
  } else if (result.defined()) {
    dstType = result.scalar_type();
  } else {
    dstType = self.scalar_type();
  }
  if (dstType == self.scalar_type()) {
    return NPUNativeFunctions::_cumprod_out(self, dim, result);
  }
  return NPUNativeFunctions::_cumprod_out(self.toType(dstType), dim, result);
}

at::Tensor& NPUNativeFunctions::cumprod_out(const at::Tensor& self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor& result) {
  at::ScalarType dstType;
  if (dtype.has_value()) {
    dstType = dtype.value();
  } else if (result.defined()) {
    dstType = result.scalar_type();
  } else {
    dstType = self.scalar_type();
    return NPUNativeFunctions::cumprod_out(self, dimname_to_position(self, dim), dstType, result);
  }
  return NPUNativeFunctions::cumprod_out(self.toType(dstType), dimname_to_position(self, dim), dstType, result);
}

at::Tensor NPUNativeFunctions::_cumprod(const at::Tensor& self, int64_t dim) {
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  NPUNativeFunctions::_cumprod_out(self, dim, result);

  return result;
}

at::Tensor NPUNativeFunctions::cumprod(const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  at::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();
  if (dstType == self.scalar_type()) {
    return NPUNativeFunctions::_cumprod(self, dim);
  }
  return NPUNativeFunctions::_cumprod(self.toType(dstType), dim);
}

at::Tensor NPUNativeFunctions::cumprod(const at::Tensor& self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  at::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();
  if (dstType == self.scalar_type()) {
    return NPUNativeFunctions::cumprod(self, dimname_to_position(self, dim), dstType);
  }
  return NPUNativeFunctions::cumprod(self, dimname_to_position(self.toType(dstType), dim), self.scalar_type());
}

at::Tensor& NPUNativeFunctions::cumprod_(at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  TORCH_CHECK(
      !dtype.has_value() || (self.scalar_type() == dtype.value()),
      "provided dtype must match the dtype of self tensor in cumprod. Got ",
      toString(self.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  at::Tensor result = OpPreparation::ApplyTensor(self);
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    NPUNativeFunctions::_cumprod_out(contiguousSelf, dim, result);
    NpuUtils::format_fresh_view(self, result);
  } else {
    NPUNativeFunctions::_cumprod_out(self, dim, result);
  }
  self.copy_(result);
  return self;
}

at::Tensor& NPUNativeFunctions::cumprod_(at::Tensor& self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  return NPUNativeFunctions::cumprod_(self, dimname_to_position(self, dim), dtype);
}

} // namespace native
} // namespace at_npu
