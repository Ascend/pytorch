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

SmallVector<int64_t, SIZE> upsample_nearest1d_npu_output_size(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales){
  SmallVector<int64_t, SIZE> outputSize;
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t W;
  if(output_size.size() != 0) {
    W = output_size[0];
  } else {
    float temp_scales = (float)scales.value();
    W = temp_scales * input.size(2);
  }
  outputSize = {N, C, W};
  return outputSize;
}

Tensor& upsample_nearest1d_out_npu(
    Tensor& result,
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales) {

  OpCommand cmd;
  cmd.Name("UpsampleNearest1d")
  
      .Input(self)
      .Output(result)
      .Attr("output_size", output_size);
      if (scales.has_value()) {
        cmd.Attr("scales", static_cast<float>(scales.value()));
      }
      cmd.Run();

  return result;
}

Tensor upsample_nearest1d_npu(
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  // calculate the output size
  SmallVector<int64_t, SIZE> outputSize = upsample_nearest1d_npu_output_size(self, output_size, scales);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  upsample_nearest1d_out_npu(result, self, output_size, scales);

  return result;
}

} // namespace native
} // namespace at
