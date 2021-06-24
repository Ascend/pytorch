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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
  namespace native {
    using namespace at::native::npu;


Tensor& softplus_out_npu(
    Tensor& result, 
    const Tensor& self, 
    Scalar beta, 
    Scalar threshold) {
  OpCommand cmd;
  cmd.Name("SoftplusV2")
      .Input(self)
      .Output(result)
      .Attr("beta", beta)
      .Attr("threshold", threshold)
      .Run();

    return result;
}

Tensor softplus_npu(
    const Tensor& self, 
    Scalar beta, 
    Scalar threshold) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  softplus_out_npu(result, self,  beta, threshold);

  return result;
}

}}  // namespace at::native