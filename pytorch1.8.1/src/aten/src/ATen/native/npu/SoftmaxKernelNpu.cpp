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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor softmax_int_npu(
    const Tensor& self,
    int64_t dim,
    optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;
    Tensor converted = dtype.has_value() ? self.to(dtype.value()) : self;
    return at::_softmax(converted, dim, false);
  }();
  namedinference::propagate_names(result, self);

  return result;
}

Tensor softmax_dimname_npu(
    const Tensor& self,
    Dimname dim,
    optional<ScalarType> dtype) {
  return softmax_int_npu(self, dimname_to_position(self, dim), dtype);
}

Tensor _softmax_npu(const Tensor& self, int64_t dim, bool half_to_float) {

  // construct the output tensor of the NPU
  Tensor result;
  if (half_to_float) {
    result = OpPreparation::ApplyTensor(self, self.options().dtype(ScalarType::Float));
  } else {
    result = OpPreparation::ApplyTensor(self);
  }

  // calculate the output result of the NPU
  optional<ScalarType> dtype = result.scalar_type();
  ScalarType dstType;
  if (dtype.has_value()) {
    dstType = dtype.value();
  } else if (result.defined()) {
    dstType = result.scalar_type();
  } else {
    dstType = self.scalar_type();
  }
  Tensor converted =
      dstType == self.scalar_type() ? self : self.to(dstType);

  SmallVector<int64_t, N> dimList = {dim};
  OpCommand cmd;
  cmd.Name("SoftmaxV2")
      .Input(converted)
      .Output(result)
      .Attr("axes", dimList)
      .Run();

  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("softmax.int", TORCH_FN(softmax_int_npu));
  m.impl("softmax.Dimname", TORCH_FN(softmax_dimname_npu));
  m.impl("_softmax", TORCH_FN(_softmax_npu));
}
} // namespace native
} // namespace at
