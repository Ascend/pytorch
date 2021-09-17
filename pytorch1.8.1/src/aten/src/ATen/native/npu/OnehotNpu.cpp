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
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUAttrDesc, N> one_hot_attr(int64_t axis, int64_t depth) {
  NPUAttrDesc npuAttrValue = NPUAttrDesc("axis", axis);
  NPUAttrDesc npuAttrValue1 = NPUAttrDesc("depth", depth);
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrValue, npuAttrValue1};

  return attrs;
}

SmallVector<NPUTensorDesc, N> one_hot_npu_output(
    const SmallVector<Tensor, N>& outputTensor) {
  return CalcuOpUtil::create_npu_output_tensor_desc(outputTensor);
}

SmallVector<NPUTensorDesc, N> one_hot_npu_input(
    const Tensor& self,
    Scalar on_value,
    Scalar off_value) {
  SmallVector<NPUTensorDesc, N> inputs;

  // auto inputTensor = CalcuOpUtil::create_npu_input_tensor_desc({self});
  // auto inputScalar =
  //    CalcuOpUtil::create_npu_input_tensor_desc({on_value, off_value},
  //    ScalarType::Float);
  Tensor on_tmp = at::empty_with_format(
                      {1},
                      self.options().dtype(ScalarType::Float),
                      CalcuOpUtil::get_tensor_npu_format(self))
                      .fill_(on_value);
  Tensor off_tmp = at::empty_with_format(
                       {1},
                       self.options().dtype(ScalarType::Float),
                       CalcuOpUtil::get_tensor_npu_format(self))
                       .fill_(off_value);
  auto inputTensor =
      CalcuOpUtil::create_npu_input_tensor_desc({self, on_tmp, off_tmp});
  inputs.insert(inputs.end(), inputTensor.begin(), inputTensor.end());
  // inputs.insert(inputs.end(), inputScalar.begin(), inputScalar.end());

  return inputs;
}

Tensor& one_hot_out_npu(
    Tensor& result,
    const Tensor& self,
    int64_t axis,
    int64_t depth,
    Scalar on_value,
    Scalar off_value) {
  auto inputs = one_hot_npu_input(self, on_value, off_value);
  auto outputs = one_hot_npu_output({result});

  auto attrs = one_hot_attr(axis, depth);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("OneHotD", inputs, outputs, attrs);

  return result;
}

Tensor one_hot_npu(
    const Tensor& self,
    int64_t axis,
    int64_t depth,
    Scalar on_value,
    Scalar off_value) {
  // calculate the output size
  auto outputSize = array_to_small_vector(self.sizes());
  outputSize.emplace_back(depth);

  Tensor result = at::empty_with_format(
      outputSize,
      self.options().dtype(ScalarType::Float),
      CalcuOpUtil::get_tensor_npu_format(self));

  one_hot_out_npu(result, self, axis, depth, on_value, off_value);

  return result;
}
} // namespace native
} // namespace at
