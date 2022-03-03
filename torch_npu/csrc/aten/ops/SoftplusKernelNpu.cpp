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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& softplus_out_nocheck(
    const at::Tensor& self,
    at::Scalar beta,
    at::Scalar threshold,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("SoftplusV2")
      .Input(self)
      .Output(result)
      .Attr("beta", beta)
      .Attr("threshold", threshold)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::softplus_out(
    const at::Tensor& self,
    at::Scalar beta,
    at::Scalar threshold,
    at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  return softplus_out_nocheck(self, beta, threshold, result);
}

at::Tensor NPUNativeFunctions::softplus(
    const at::Tensor& self,
    at::Scalar beta,
    at::Scalar threshold) {
  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(
      outputSize, self.options(), self);
  softplus_out_nocheck(self, beta, threshold, result);
  return result;
}

} // namespace native
} // namespace at_npu