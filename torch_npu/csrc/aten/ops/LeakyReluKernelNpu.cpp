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

at::Tensor& leaky_relu_out_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar negval) {
  OpCommand cmd;
  cmd.Name("LeakyRelu")
      .Input(self)
      .Output(result)
      .Attr("negative_slope", negval)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::leaky_relu_out(const at::Tensor& self, const at::Scalar& negval, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    at::Tensor newResult = leaky_relu_out_nocheck(contiguousResult, self, negval);
    NpuUtils::format_fresh_view(result, newResult);
  } else {
    leaky_relu_out_nocheck(result, self, negval);
  }

  return result;
}

at::Tensor NPUNativeFunctions::leaky_relu(const at::Tensor& self, const at::Scalar& negval) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  leaky_relu_out_nocheck(result, self, negval);
  return result;
}

at::Tensor& NPUNativeFunctions::leaky_relu_(at::Tensor& self, const at::Scalar& neg_val) {
  NPUNativeFunctions::leaky_relu_out(self, neg_val, self);

  return self;
}

} // namespace native
} // namespace at_npu