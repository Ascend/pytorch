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
#include "c10/npu/OptionsManager.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& repeat_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef repeats) {
  int value = repeats.size();
  vector<int> tmp_vector = {};
  for (int i = 0; i < value; i++){
    tmp_vector.emplace_back(repeats[i]);
  }

  OpCommand cmd;
  cmd.Name("Tile")
    .Input(self)
    .Input(repeats)
    .Output(result)
    .Run();
  return result;
}

Tensor repeat_npu(const Tensor& self, IntArrayRef repeats) {
  TORCH_CHECK(repeats.size() >= self.ndimension(),
              "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");
  Tensor selfCp = self;
  if(repeats.size() > selfCp.ndimension()){
    auto diff = repeats.size() - selfCp.ndimension();
    for(int i=0;i<diff;i++){
      selfCp = at::unsqueeze(selfCp, 0);
    }
  }

  // calculate the output size
  auto outputSize = repeat_npu_output_size(selfCp, repeats);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, selfCp.options(), CalcuOpUtil::get_tensor_npu_format(selfCp));

  // calculate the output result of the NPU
  repeat_out_npu(result, selfCp, repeats);
  return result;
}

} // namespace native
} // namespace at