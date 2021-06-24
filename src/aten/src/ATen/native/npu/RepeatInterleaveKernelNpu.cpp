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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> repeat_interleave_npu_input(
    const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> repeat_interleave_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUAttrDesc, N> repeat_interleave_npu_attr(int64_t repeats, int64_t dim) {
  NPUAttrDesc npuAttrRepeats = NPUAttrDesc("tiles", repeats);
  NPUAttrDesc npuAttrDim = NPUAttrDesc("axis", dim);

  SmallVector<NPUAttrDesc, N> attrs = {npuAttrRepeats, npuAttrDim};
  return attrs;
}

Tensor& repeat_interleave_out_npu(Tensor& result, Tensor& self, int64_t repeats, int64_t dim) {
  // constructs the input and output NPUTensorDesc
  auto inputs = repeat_interleave_npu_input({self});
  auto outputs = repeat_interleave_npu_output({result});

  // constructs the attr of the NPUAttrDesc
  auto attrs = repeat_interleave_npu_attr(repeats, dim);
  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("TileWithAxis", inputs, outputs, attrs);

  return result;
}

Tensor repeat_interleave_npu(const Tensor &self, int64_t repeats, c10::optional<int64_t> dim) {
  int64_t realDim = dim.value_or(0);
  
  //dim value must be greater than or equal to 0.
  int64_t self_dim = self.dim();
  if((realDim < -self_dim) || (realDim > self_dim - 1)){
    AT_ERROR("dim value should be in the range of [-x, x-1], x is the dimension number of input tensor.");
  }

  Tensor selfTensor = self;
  if(!dim.has_value()){
    selfTensor = at::flatten(selfTensor);
  }
  // calculate the output size
  auto outputSize = repeat_interleave_npu_output_size(selfTensor, repeats, realDim);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      selfTensor.options(),
      CalcuOpUtil::get_tensor_npu_format(selfTensor));

  // calculate the output result of the NPU
  repeat_interleave_out_npu(result, selfTensor, repeats, realDim);

  return result;
}

} // namespace native
} // namespace at