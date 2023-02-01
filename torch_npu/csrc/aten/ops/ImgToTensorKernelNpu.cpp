// Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor &img_to_tensor_out(const at::Tensor &self, at::Tensor &result)
{
  OpCommand cmd;
  cmd.Name("ImgToTensor")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::img_to_tensor(const at::Tensor &self)
{
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(at::kFloat),
      CalcuOpUtil::GetTensorNpuFormat(self));

  // calculate the output result of the NPU
  img_to_tensor_out(self, result);

  return result;
}

} // namespace native
} // namespace at_npu