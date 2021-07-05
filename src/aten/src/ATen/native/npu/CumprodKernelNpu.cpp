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

#include "ATen/native/npu/utils/OpAdapter.h"
#include<ATen/NamedTensorUtils.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& _cumprod_out_npu(Tensor& result, const Tensor& self, int64_t dim) {
  OpCommand cmd;
  cmd.Name("Cumprod")
    .Input(self)
    .Output(result)
    .Attr("axis", dim)
    .Run();

  return result;
}

Tensor& cumprod_out_npu(Tensor& result, const Tensor& self, int64_t dim, optional<ScalarType> dtype) {
  ScalarType dstType;
  if (dtype.has_value()) {
    dstType = dtype.value();
  } else if (result.defined()) {
    dstType = result.scalar_type();
  } else {
    dstType = self.scalar_type();
  }
  if (dstType == self.scalar_type()) {
    return at::_cumprod_out(result, self, dim);
  }
  return at::_cumprod_out(result, self.toType(dstType), dim);
}

Tensor& cumprod_out_npu(Tensor& result, const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  ScalarType dstType;
  if (dtype.has_value()) {
    dstType = dtype.value();
  } else if (result.defined()) {
    dstType = result.scalar_type();
  } else {
    dstType = self.scalar_type();
    return at::cumprod_out(result, self, dimname_to_position(self, dim));
  }
  return at::cumprod_out(result, self.toType(dstType), dimname_to_position(self, dim));
}

Tensor _cumprod_npu(const Tensor& self, int64_t dim) {
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  _cumprod_out_npu(result, self, dim);

  return result;
}

Tensor cumprod_npu(const Tensor& self, int64_t dim, optional<ScalarType> dtype) {
  ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();
  if (dstType == self.scalar_type()) {
    return at::_cumprod(self, dim);
  }
  return at::_cumprod(self.toType(dstType), dim);
}

Tensor cumprod_npu(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();
  if (dstType == self.scalar_type()) {
    return at::cumprod(self, dimname_to_position(self, dim));
  }
  return at::cumprod(self, dimname_to_position(self.toType(dstType), dim));
}

} // namespace native
} // namespace at
