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

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
at::Tensor& ps_roi_pooling_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
  OpCommand cmd;
  cmd.Name("PSROIPoolingV2")
      .Input(self, "x", ACL_FORMAT_NCHW)
      .Input(rois)
      .Output(result, "y", ACL_FORMAT_NCHW)
      .Attr("spatial_scale", (float)spatial_scale)
      .Attr("output_dim", output_dim)
      .Attr("group_size", group_size)
      .Run();

  return result;
}

at::Tensor ps_roi_pooling(
    const at::Tensor& self,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
  auto outputSize ={
      rois.size(0) * rois.size(2), output_dim, group_size, group_size};

  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  ps_roi_pooling_npu_nocheck(
      result,
      self,
      rois,
      spatial_scale,
      group_size,
      output_dim);

  return result;
}

at::Tensor& ps_roi_pooling_backward_npu_nocheck(
    at::Tensor& input_grad,
    const at::Tensor& output_grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim,
    at::IntArrayRef input_size) {
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

at::Tensor NPUNativeFunctions::npu_ps_roi_pooling_backward(
    const at::Tensor& output_grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim,
    at::IntArrayRef input_size) {
  auto outputSize ={
      rois.size(0), group_size * group_size * output_dim, input_size[0], input_size[1]};

  at::Tensor input_grad = OpPreparation::ApplyTensor(output_grad, outputSize);

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

class NPUPsRoiPoolingFunction: public torch::autograd::Function<NPUPsRoiPoolingFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
    const at::Tensor& self,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["group_size"] = group_size;
    ctx->saved_data["output_dim"] = output_dim;
    c10::SmallVector<int64_t, N> input_size_vec = {self.size(2), self.size(3)};
    at::IntArrayRef input_size(input_size_vec);
    ctx->saved_data["input_size"] = input_size;
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self, rois});
    return ps_roi_pooling(self, rois, spatial_scale, group_size, output_dim);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto spatial_scale = ctx->saved_data["spatial_scale"].toDouble();
    auto group_size = ctx->saved_data["group_size"].toInt();
    auto output_dim = ctx->saved_data["output_dim"].toInt();
    auto input_size = ctx->saved_data["input_size"].toIntVector();
    auto saved = ctx->get_saved_variables();
    auto self = saved[0];
    auto rois = saved[1];


    at::Tensor result = NPUNativeFunctions::npu_ps_roi_pooling_backward(grad_outputs[0],
        rois,
        spatial_scale,
        group_size,
        output_dim,
        input_size);
    tensor_list output = {result,
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
    return output;
  }
};

at::Tensor NPUNativeFunctions::npu_ps_roi_pooling(const at::Tensor& self,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
    return NPUPsRoiPoolingFunction::apply(self, rois, spatial_scale, group_size, output_dim);
}

} // namespace native
} // namespace at_npu