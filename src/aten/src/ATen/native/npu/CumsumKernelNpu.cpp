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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& _cumsum_out_npu(Tensor& result, const Tensor& self, int64_t dim) {
  OpCommand cmd;
  if (!c10::npu::OptionsManager::CheckDynamicEnable()) {
    if (dim == 0) {
      Scalar dimScalar(dim);
      cmd.Name("Cumsum")
      .Input(self)
      .Input(dimScalar, at::kLong)
      .Output(result)
      .Run();
    } else {
      SmallVector<int64_t, N> dimVec = {dim};
      cmd.Name("Cumsum")
      .Input(self)
      .Input(dimVec, at::kInt)
      .Output(result)
      .Run();
    }
  } else {
    cmd.Name("CumsumD")
      .Input(self)
      .Output(result)
      .Attr("axis", dim)
      .Run();
  }

  return result;
}

Tensor _cumsum_npu(const Tensor& self, int64_t dim) {
  Tensor input = self;
  if(self.scalar_type() == ScalarType::Long){
     input = self.to(ScalarType::Int);
  }
  // calculate the output size
  auto outputSize = input_same_output_size(self);
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(input, outputSize);
  // calculate the output result of the NPU
  _cumsum_out_npu(result, input, dim);

  return result;
}

} // namespace native
} // namespace at