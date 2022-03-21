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

tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> lstm_cell_npu(
    const Tensor &_input,
    const Tensor &w_ih,
    const Tensor &w_hh,
    const Tensor &bias,
    const Tensor &_h,
    const Tensor &_c) {
  Tensor input = _input.reshape({1, _input.size(0), _input.size(1)});
  Tensor h = _h.reshape({1, _h.size(0), _h.size(1)});
  Tensor c = _c.reshape({1, _c.size(0), _c.size(1)});
  int64_t numStep = input.size(0);
  int64_t batchSize = input.size(1);
  int64_t hiddenSize = bias.size(0) / 4;

  SmallVector<int64_t, 8> outputSize = {numStep, batchSize, hiddenSize};
  Tensor yOutput = OpPreparation::ApplyTensor(input, outputSize);
  Tensor hOutput = OpPreparation::ApplyTensor(input, outputSize);
  Tensor cOutput = OpPreparation::ApplyTensor(input, outputSize);
  Tensor iOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  Tensor jOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  Tensor fOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  Tensor oOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  Tensor tanhc = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);

  OpCommand cmd;
  cmd.Name("DynamicRNNV2")
      .Input(input)
      .Input(w_ih)
      .Input(w_hh)
      .Input(bias)
      .Input()
      .Input(h)
      .Input(c)
      .Output(yOutput)
      .Output(hOutput)
      .Output(cOutput)
      .Output(iOutput)
      .Output(jOutput)
      .Output(fOutput)
      .Output(oOutput)
      .Output(tanhc)
      .Attr("cell_type", (string)"LSTM")
      .Attr("direction", (string)"UNIDIRECTIONAL")
      .Attr("cell_depth", (int64_t)1)
      .Attr("use_peephole", (bool)false)
      .Attr("keep_prob", (float)1.0)
      .Attr("cell_clip", (float)-1.0)
      .Attr("num_proj", (int64_t)0)
      .Attr("time_major", (bool)true)
      .Attr("activation", (string)"tanh")
      .Attr("forget_bias", (float)0.0)
      .Attr("gate_order", (string)"ifco")
      .Run();

  Tensor hOut = hOutput[0];
  Tensor cOut = cOutput[0];
  return std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(
      yOutput, hOut, cOut, iOutput, jOutput, fOutput, oOutput, tanhc);
}

tuple<Tensor, Tensor> lstm_cell_npu(
    const Tensor &input,
    TensorList hx,
    const Tensor &w_ih,
    const Tensor &w_hh,
    const Tensor &b_ih,
    const Tensor &b_hh) {
  Tensor weight_ih = w_ih.t().to(input.dtype());
  Tensor weight_hh = w_hh.t().to(input.dtype());
  Tensor bias = at::add(b_ih, b_hh).to(input.dtype());
  Tensor h = hx[0];
  Tensor c = hx[1];

  auto results = at::npu_lstm_cell(input, weight_ih, weight_hh, bias, h, c);
  return std::tuple<Tensor, Tensor>(std::get<1>(results), std::get<2>(results));
}
} // namespace native
} // namespace at
