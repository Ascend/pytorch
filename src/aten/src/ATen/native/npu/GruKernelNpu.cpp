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

tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> gru_npu(
    const Tensor& input,
    const Tensor& hx,
    const Tensor& weight_input,
    const Tensor& weight_hidden,
    const Tensor& bias_input,
    const Tensor& bias_hidden,
    const Tensor& seq_length,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  int64_t numStep = input.size(0);
  int64_t batchSize = input.size(1);
  int64_t hiddenSize = bias_input.size(0) / 3;
  SmallVector<int64_t, SIZE> outputSize = {numStep, batchSize, hiddenSize};
  int64_t npu_format = ACL_FORMAT_FRACTAL_NZ;

  Tensor output_y = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      bias_input.options(),
      npu_format);
  Tensor output_h = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      bias_input.options(),
      ACL_FORMAT_ND); // 后续需要做slice和unsqueeze，BaseFormat方便
  Tensor output_updata = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      bias_input.options(),
      npu_format);
  Tensor output_reset = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      bias_input.options(),
      npu_format);
  Tensor output_new = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      bias_input.options(),
      npu_format);
  Tensor hidden_new = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      bias_input.options(),
      npu_format);
  

  OpCommand cmd;
  cmd.Name("DynamicGRUV2")
      .Input(input)
      .Input(weight_input)
      .Input(weight_hidden)
      .Input(bias_input)
      .Input(bias_hidden)
      .Input()
      .Input(hx)
      .Output(output_y)
      .Output(output_h)
      .Output(output_updata)
      .Output(output_reset)
      .Output(output_new)
      .Output(hidden_new)
      .Attr("direction", (string)"UNIDIRECTIONAL")
      .Attr("cell_depth", (int64_t)1)
      .Attr("keep_prob", (float)1.0)
      .Attr("cell_clip", (float)-1.0)
      .Attr("num_proj", (int64_t)0)
      .Attr("time_major", true)
      .Attr("activation", (string)"tanh")
      .Attr("gate_order", (string)"rzh")
      .Attr("reset_after", true)
      .Attr("is_training", true)
      .Run();

  return std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(
      output_y, output_h, output_updata, output_reset, output_new, hidden_new);
}

tuple<Tensor, Tensor> gru_npu_(
    const Tensor& input,
    const Tensor& hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  // get weight  fp16
  Tensor weight_input = params[0].t();
  Tensor weight_hidden = params[1].t();

  // get bias  fp16 / fp32
  Tensor bias_input;
  Tensor bias_hidden;
  if (has_biases) {
    bias_input = params[2].to(input.dtype());
    bias_hidden = params[3].to(input.dtype());
  } else {
    bias_input = OpPreparation::ApplyTensorWithFormat(weight_input.size(0), input.options(), ACL_FORMAT_FRACTAL_NZ).mul(0);
    bias_hidden = OpPreparation::ApplyTensorWithFormat(weight_hidden.size(0), input.options(), ACL_FORMAT_FRACTAL_NZ).mul(0);
  }

  Tensor seq_length = OpPreparation::ApplyTensorWithFormat({}, input.options(), ACL_FORMAT_ND);

  auto results = at::npu_gru(
      input,
      hx,
      weight_input,
      weight_hidden,
      bias_input,
      bias_hidden,
      seq_length,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first);
  int64_t numStep = input.size(0);
  Tensor output_hy = at::unsqueeze(std::get<1>(results)[numStep - 1], 0);

  return std::tuple<Tensor, Tensor>(std::get<0>(results), output_hy);
}

} // namespace native
} // namespace at
