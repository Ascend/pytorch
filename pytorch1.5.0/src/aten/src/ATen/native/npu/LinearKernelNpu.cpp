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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor linear_npu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  SmallVector<int64_t, SIZE> outputSize = {input.size(0), weight.size(0)};
  Tensor output = OpPreparation::ApplyTensor(input, outputSize);

  int64_t offset_x = 0;
  OpCommand cmd;
  cmd.Name("MatMulV2")
      .Input(input)
      .Input(weight);
  if (bias.defined()) {
    cmd.Input(bias);
  }
  cmd.Output(output)
      .Attr("transpose_x1", false)
      .Attr("transpose_x2", true)
      .Attr("offset_x", offset_x)
      .Run();
  
  return output;
}

} // namespace native
} // namespace at
