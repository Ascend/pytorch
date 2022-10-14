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

at::Tensor& repeat_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef repeats) {
  
  OpCommand cmd;
  cmd.Name("Tile")
    .Input(self)
    .Input(repeats)
    .Output(result)
    .Run();

  return result;
}

at::Tensor NPUNativeFunctions::repeat(const at::Tensor& self, at::IntArrayRef repeats) {
  TORCH_CHECK(repeats.size() >= self.ndimension(),
              "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");
  at::Tensor selfCp = self;
  if(repeats.size() > selfCp.ndimension()){
    auto diff = repeats.size() - selfCp.ndimension();
    for(int i=0;i<diff;i++){
      selfCp = at::unsqueeze(selfCp, 0);
    }
  }

  // calculate the output size
  auto outputSize = repeat_npu_output_size(selfCp, repeats);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, selfCp.options(), CalcuOpUtil::get_tensor_npu_format(selfCp));

  // calculate the output result of the NPU
  repeat_out_npu_nocheck(result, selfCp, repeats);
  return result;
}
} // namespace native
} // namespace at_npu