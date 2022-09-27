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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> slogdet_out_nocheck_npu(
    const at::Tensor& self,
    at::Tensor& sign,
    at::Tensor& y) {
  OpCommand cmd;
  cmd.Name("LogMatrixDeterminant")
      .Input(self)
      .Output(sign)
      .Output(y)
      .Run();

  return std::tie(sign, y);
}

tuple<at::Tensor, at::Tensor>NPUNativeFunctions::slogdet(const at::Tensor& self) {

  TORCH_CHECK(self.dim() >= 2, "input must be at least 2 dimensions");

  // calculate the output size
  auto outputSize = array_to_small_vector(self.sizes());
  outputSize.erase(outputSize.end() - 2, outputSize.end());

  // construct the output tensor of the NPU
  at::Tensor sign = OpPreparation::ApplyTensor(self, outputSize);
  at::Tensor y = OpPreparation::ApplyTensor(self, outputSize);
  
  // calculate the output result of the NPU
  slogdet_out_nocheck_npu(self, sign, y);

  return std::tie(sign, y);
}

} // namespace native
} // namespace at_npu
