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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> gru_backward_npu(
    const Tensor& grady,
    const Tensor& gradh,
    const Tensor& input,
    const Tensor& weight_input,
    const Tensor& weight_hidden,
    const Tensor& bias_input,
    const Tensor& bias_hidden,
    const Tensor& seq_length,
    const Tensor& init_h,
    const Tensor& output_y,
    const Tensor& output_h,
    const Tensor& output_updata,
    const Tensor& output_reset,
    const Tensor& output_new,
    const Tensor& hidden_new) {
 
  Tensor inh = at::squeeze(init_h, 0);
  auto grad_y =
      grady.defined() ? grady : OpPreparation::ApplyTensorWithFormat(output_y.sizes(), output_y.options(), ACL_FORMAT_FRACTAL_NZ).mul(0);
  auto grad_h =
      gradh.defined() ? gradh[input.size(0)-1] : OpPreparation::ApplyTensorWithFormat(inh.sizes(), output_h.options(), ACL_FORMAT_FRACTAL_NZ).mul(0);

  Tensor mask = at::zeros({}, input.options().dtype(kByte)); // uint8
  Tensor seq_lengths = at::zeros({}, input.options());

  int64_t npu_format = ACL_FORMAT_ND;

  Tensor grad_w_input = OpPreparation::ApplyTensorWithFormat(weight_input.sizes(), input.options(), npu_format);
  Tensor grad_w_hidden = OpPreparation::ApplyTensorWithFormat(weight_hidden.sizes(), input.options(), npu_format);
  Tensor grad_x = OpPreparation::ApplyTensorWithFormat(input.sizes(), input.options(), npu_format);
  Tensor grad_b_input = OpPreparation::ApplyTensorWithFormat(bias_input.sizes(), input.options(), npu_format);
  Tensor grad_b_hidden = OpPreparation::ApplyTensorWithFormat(bias_hidden.sizes(), input.options(), npu_format);
  Tensor grad_h_prev = OpPreparation::ApplyTensorWithFormat(init_h.sizes(), input.options(), npu_format);

  OpCommand cmd;
  cmd.Name("DynamicGRUV2Grad")
      .Input(input)
      .Input(weight_input)
      .Input(weight_hidden)
      .Input(output_y)
      .Input(inh)
      .Input(output_h)
      .Input(grad_y)
      .Input(grad_h)
      .Input(output_updata)
      .Input(output_reset)
      .Input(output_new)
      .Input(hidden_new)
      .Input(seq_lengths)
      .Input(mask)
      .Output(grad_w_input)
      .Output(grad_w_hidden)
      .Output(grad_b_input)
      .Output(grad_b_hidden)
      .Output(grad_x)
      .Output(grad_h_prev)
      .Attr("direction", (string) "UNIDIRECTIONAL")
      .Attr("cell_depth", (int64_t)1)
      .Attr("keep_prob", (float)1.0)
      .Attr("cell_clip", (float)-1.0)
      .Attr("num_proj", (int64_t)0)
      .Attr("time_major", (bool)true)
      .Attr("bias_type", (string) "no_bias")
      .Attr("gate_order", (string) "rzh")
      .Attr("reset_after", (bool)true)
      .Run();

  return std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> {
      grad_w_input, grad_w_hidden, grad_x, grad_b_input, grad_b_hidden, grad_h_prev
  };
}

} // namespace native
} // namespace at