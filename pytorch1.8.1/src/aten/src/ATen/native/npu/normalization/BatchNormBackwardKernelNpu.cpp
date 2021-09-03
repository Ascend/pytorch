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

#include <c10/npu/NPUCachingAllocator.h>
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

namespace{
tuple<Tensor&, Tensor&> batch_norm_backward_training_update_nocheck(
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_out,
    const Tensor& self,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps) {
  OpCommand cmd;
  cmd.Name("BNTrainingUpdateGrad")
      .Input(grad_out)
      .Input(self)
      .Input(save_mean)
      .Input(save_invstd)
      .Output(grad_weight)
      .Output(grad_bias)
      .Attr("epsilon", static_cast<float>(eps))
      .Run();
  
  return tuple<Tensor&, Tensor&>(grad_weight, grad_bias);
}

Tensor& batch_norm_backward_training_reduce_nocheck(
    Tensor& grad_input,
    const Tensor& grad_weight,
    const Tensor& grad_bias,
    const Tensor& grad_out,
    const Tensor& self,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps) {
  OpCommand cmd;
  cmd.Name("BNTrainingReduceGrad")
      .Input(grad_out)
      .Input(self)
      .Input(grad_weight)
      .Input(grad_bias)
      .Input(weight)
      .Input(save_mean)
      .Input(save_invstd)
      .Output(grad_input)
      .Attr("epsilon", static_cast<float>(eps))
      .Run();

  return grad_input;
}

Tensor& batch_norm_backward_infer_nocheck(
    Tensor& grad_input,
    const Tensor& grad_weight,
    const Tensor& grad_bias,
    const Tensor& grad_out,
    const Tensor& self,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps)  {
  OpCommand cmd;
  cmd.Name("BNInferGrad")
      .Input(grad_out)
      .Input(weight)
      .Input(running_var)
      .Output(grad_input)
      .Attr("epsilon", static_cast<float>(eps))
      .Run();
  
  return grad_input;  
}

tuple<Tensor&, Tensor&, Tensor&> batch_norm_backward_impl(
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_out,
    const Tensor& self,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3> grad_input_mask) {
  // note: when not train, save_mean/save_invstd replaced by running_mean/running_var
  Tensor mean = train ? save_mean : running_mean;
  Tensor invstd = train ? save_invstd : running_var;

  batch_norm_backward_training_update_nocheck(
      grad_weight,
      grad_bias,
      grad_out,
      self,
      weight,
      running_mean,
      running_var,
      mean,
      invstd,
      train,
      eps);

  // calculate grad_input by NPU 
  if (grad_input_mask[0]) {
    if (!train) {
      batch_norm_backward_infer_nocheck(
          grad_input,
          grad_weight,
          grad_bias,
          grad_out,
          self,
          weight,
          running_mean,
          running_var,
          mean,
          invstd,
          train,
          eps);
    } else {
      batch_norm_backward_training_reduce_nocheck(
          grad_input,
          grad_weight,
          grad_bias,
          grad_out,
          self,
          weight,
          running_mean,
          running_var,
          mean,
          invstd,
          train,
          eps);
    }
  }
  
  return tuple<Tensor&, Tensor&, Tensor&>(grad_input, grad_weight, grad_bias);
}
}

tuple<Tensor, Tensor, Tensor> batch_norm_backward_npu(
    const Tensor& grad_out,
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    const c10::optional<Tensor>& save_mean_opt,
    const c10::optional<Tensor>& save_invstd_opt,
    bool train,
    double eps,
    std::array<bool, 3> grad_input_mask) {

  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  const Tensor& save_mean = c10::value_or_else(save_mean_opt, [] {return Tensor();});
  const Tensor& save_invstd = c10::value_or_else(save_invstd_opt, [] {return Tensor();});

  Tensor self_4d;
  Tensor grad_out_4d;
  SmallVector<int64_t, N> self_shape = array_to_small_vector(self.sizes());

  if (grad_out.dim() <= 4) {
    SmallVector<int64_t, N> nchw_shape(self_shape);
    nchw_shape.resize(4, 1);
    self_4d = self.reshape(nchw_shape);
    grad_out_4d = grad_out.reshape(nchw_shape);
  } else if (grad_out.dim() == 5) {
    // ncdhw -> ndchw
    self_4d = self.permute({0, 2, 1, 3, 4});
    grad_out_4d = grad_out.permute({0, 2, 1, 3, 4});
    // nchw=(n*d, c, h, w)
    SmallVector<int64_t, N> nchw_shape = {self_shape[0] * self_shape[2], self_shape[1], self_shape[3], self_shape[4]};
    // ndchw -> nchw
    self_4d = self_4d.reshape(nchw_shape);
    grad_out_4d = grad_out_4d.reshape(nchw_shape);
  }

  // init optional input
  int64_t dim_c = self_4d.size(1);
  TensorOptions options = self.options().dtype(ScalarType::Float);

  Tensor weight_tensor = weight.defined() ? weight : at::ones({dim_c}, options);
  Tensor running_mean_tensor = running_mean.defined() ? running_mean : at::zeros({dim_c}, options);
  Tensor running_var_tensor = running_var.defined() ? running_var : at::ones({dim_c}, options);

  // construct the output tensor of the NPU
  Tensor grad_input = OpPreparation::ApplyTensor(self_4d.sizes(), self_4d.options(), self_4d);
  Tensor grad_weight = OpPreparation::ApplyTensor(weight_tensor.sizes(), weight_tensor.options(), weight_tensor);
  Tensor grad_bias = OpPreparation::ApplyTensor(weight_tensor.sizes(), weight_tensor.options(), weight_tensor);

  // calculate the output result of the NPU
  batch_norm_backward_impl(
      grad_input,
      grad_weight,
      grad_bias,
      grad_out_4d,
      self_4d,
      weight_tensor,
      running_mean_tensor,
      running_var_tensor,
      save_mean,
      save_invstd,
      train,
      eps,
      grad_input_mask);
  
  // grad_input_mask
  Tensor undefine_grad_input;
  Tensor undefine_grad_weight;
  Tensor undefine_grad_bias;

  if (grad_input_mask[0]) {
    if (self.dim() == 5) {
      //NCHW -> NDCHW ->NCDHW
      std::swap(self_shape[1], self_shape[2]);
      grad_input = grad_input.view(self_shape);
      grad_input = NpuUtils::format_contiguous(grad_input);
      grad_input = grad_input.permute({0, 2, 1, 3, 4}).clone();
    } else if (self.dim() < 5) {
      grad_input = grad_input.view(self_shape);
      grad_input = NpuUtils::format_contiguous(grad_input);
    }
  } else {
    grad_input = undefine_grad_input;
  }

  if (!grad_input_mask[1]) {
    grad_weight = undefine_grad_weight;
  }

  if (!grad_input_mask[2]) {
    grad_bias = undefine_grad_bias;
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("native_batch_norm_backward", TORCH_FN(batch_norm_backward_npu));
}

} // namespace native
} // namespace at
