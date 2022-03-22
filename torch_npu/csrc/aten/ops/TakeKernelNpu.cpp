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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& take_out_nocheck(const at::Tensor& self, const at::Tensor& index, at::Tensor& result) {
  at::Tensor input_tensor = self.reshape(-1);
  at::Tensor contiguousSelf = NpuUtils::format_contiguous(input_tensor);
  at::Tensor contiguousIndex = NpuUtils::format_contiguous(index);

  OpCommand cmd;
  cmd.Name("Gather")
      .Input(contiguousSelf)
      .Input(contiguousIndex)
      .Output(result)
      .Attr("validate_indices", false)
      .Run();
  
  return result;
}

at::Tensor& NPUNativeFunctions::take_out(const at::Tensor& self, const at::Tensor& index, at::Tensor& result) {
  // calculate the output size
  auto outputSize = input_same_output_size(index);

  OpPreparation::CheckOut(
      {self, index},
      result,
      self,
      outputSize);

  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    take_out_nocheck(self, index, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    take_out_nocheck(self, index, result);
  }

  return result;
}

at::Tensor NPUNativeFunctions::take(const at::Tensor& self, const at::Tensor& index) {
  // calculate the output size
  auto outputSize = input_same_output_size(index);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithSizes(
      outputSize,
      self.options());

  take_out_nocheck(self, index, result);
  
  return result;
}
} // namespace native
} // namespace at_npu