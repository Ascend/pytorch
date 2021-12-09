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
#include <torch/csrc/autograd/custom_function.h>

using namespace torch::autograd;

namespace at {
namespace native {
using namespace at::native::npu;

Tensor ps_roi_pooling_npu(
    const Tensor& self,
    const Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim);

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
      .Input(output_grad)
      .Input(rois)
      .Output(input_grad)
      .Attr("spatial_scale", (float)spatial_scale)
      .Attr("group_size", group_size)
      .Attr("output_dim", output_dim)
      .Attr("input_size", input_size)
      .Run();

  return input_grad;
}

Tensor ps_roi_pooling_backward_npu(
    const Tensor& output_grad,
    const Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim,
    IntArrayRef input_size) {
  auto outputSize ={
      rois.size(0), group_size * group_size * output_dim, input_size[0], input_size[1]};

  Tensor input_grad = OpPreparation::ApplyTensor(output_grad, outputSize);

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
  static Tensor forward(AutogradContext *ctx,
    const Tensor& self,
    const Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
    ctx->saved_data["spatial_scale"] = spatial_scale;
    ctx->saved_data["group_size"] = group_size;
    ctx->saved_data["output_dim"] = output_dim;
    ctx->saved_data["rois"] = rois;
    SmallVector<int64_t, N> input_size_vec = {self.size(2), self.size(3)};
    IntArrayRef input_size(input_size_vec);
    ctx->saved_data["input_size"] = input_size;
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self});
    return ps_roi_pooling_npu(self, rois, spatial_scale, group_size, output_dim);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto spatial_scale = ctx->saved_data["spatial_scale"].toDouble();
    auto group_size = ctx->saved_data["group_size"].toInt();
    auto output_dim = ctx->saved_data["output_dim"].toInt();
    auto input_size = ctx->saved_data["input_size"].toIntVector();
    auto rois = ctx->saved_data["rois"].toTensor();
    auto saved = ctx->get_saved_variables();
    auto self = saved[0];

    Tensor result = at::native::ps_roi_pooling_backward_npu(grad_outputs[0],
        rois,
        spatial_scale,
        group_size,
        output_dim,
        input_size);
    tensor_list output = {result,
        Tensor(),
        Tensor(),
        Tensor(),
        Tensor()};
    return output;
  }
};

Tensor npu_ps_roi_pooling_autograd(const Tensor& self,
    const Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim) {
    return NPUPsRoiPoolingFunction::apply(self, rois, spatial_scale, group_size, output_dim);
}

TORCH_LIBRARY_IMPL(aten, AutogradNPU, m) {
    m.impl("npu_ps_roi_pooling", npu_ps_roi_pooling_autograd);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_ps_roi_pooling_backward", TORCH_FN(ps_roi_pooling_backward_npu));
}

} // namespace native
} // namespace at