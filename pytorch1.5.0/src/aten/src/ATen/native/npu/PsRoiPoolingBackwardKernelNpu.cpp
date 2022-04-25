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
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

namespace {
Tensor& ps_roi_pooling_backward_npu_nocheck(
    Tensor& input_grad,
    const Tensor& output_grad,
    const Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim,
    IntArrayRef input_size) {
  OpCommand cmd;
  cmd.Name("PSROIPoolingGradV2D")
      .Input(output_grad, "x", ACL_FORMAT_NCHW)
      .Input(rois)
      .Output(input_grad, "y", ACL_FORMAT_NCHW)
      .Attr("spatial_scale", (float)spatial_scale)
      .Attr("group_size", group_size)
      .Attr("output_dim", output_dim)
      .Attr("input_size", input_size)
      .Run();

  return input_grad;
}
} // namespace

Tensor ps_roi_pooling_backward_npu(
    const Tensor& output_grad,
    const Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim,
    IntArrayRef input_size) {
  // caculate outputsize
  auto outputSize ={
      rois.size(0), group_size * group_size * output_dim, input_size[0], input_size[1]};  

  // construct the output tensor of the NPU
  Tensor input_grad = 
      at::empty_with_format(outputSize, output_grad.options(), CalcuOpUtil::get_tensor_npu_format(output_grad));

  // calculate the output result of the NPU
  ps_roi_pooling_backward_npu_nocheck(
      input_grad,
      output_grad,
      rois,
      spatial_scale,
      group_size,
      output_dim,
      input_size);

  return input_grad;
}

} // namespace native
} // namespace at