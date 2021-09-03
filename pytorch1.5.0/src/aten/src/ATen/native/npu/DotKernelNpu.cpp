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

Tensor& dot_out_npu(Tensor& result, const Tensor& self, const Tensor& tensor) {
  // constructs the input and output NPUTensorDesc     
  OpCommand cmd;
  cmd.Name("Dot")
      .Input(self)
      .Input(tensor)
      .Output(result)
      .Run();
  SmallVector<int64_t, N> shape = {};
  result.resize_(shape);
  return result;
}
Tensor dot_npu(const Tensor& self, const Tensor& tensor) {
  SmallVector<int64_t, SIZE> outputSize = dot_npu_output_size(self, tensor);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  dot_out_npu(result, self, tensor);
  return result;
}
} // namespace native
} // namespace at