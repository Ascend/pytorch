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

at::Tensor &crop_and_resize_out(
    const at::Tensor &self,
    const at::Tensor &boxes,
    const at::Tensor &box_index,
    at::IntArrayRef crop_size,
    double extrapolation_value,
    c10::string_view method,
    at::Tensor &result)
{
  OpCommand cmd;
  cmd.Name("CropAndResize")
      .Input(self)
      .Input(boxes)
      .Input(box_index)
      .Input(crop_size, at::kInt)
      .Output(result)
      .Attr<float>("extrapolation_value", extrapolation_value)
      .Attr<std::string>("method", std::string(method).data())
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::crop_and_resize(
    const at::Tensor &self,
    const at::Tensor &boxes,
    const at::Tensor &box_index,
    at::IntArrayRef crop_size,
    double extrapolation_value,
    c10::string_view method)
{
  // calculate the output size
  auto outputSize = crop_and_resize_npu_output_size(self, boxes, crop_size);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options().dtype(boxes.dtype()),
      CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  crop_and_resize_out(
      self,
      boxes, box_index, crop_size,
      extrapolation_value, method,
      result);

  return result;
}

} // namespace native
} // namespace at_npu