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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

struct CellParams {
  CellParams(const Tensor& _w_ih, const Tensor& _w_hh)
    : w_ih(_w_ih), w_hh(_w_hh), b_ih({}), b_hh({}) {};
  CellParams(const Tensor& _w_ih, const Tensor& _w_hh, const Tensor& _b_ih, const Tensor& _b_hh)
    : w_ih(_w_ih), w_hh(_w_hh), b_ih(_b_ih), b_hh(_b_hh) {};
  const Tensor& w_ih;
  const Tensor& w_hh;
  const Tensor& b_ih; /* optional */
  const Tensor& b_hh; /* optional */
};

using BidirectCellParams = std::pair<CellParams, CellParams>;
using pair_of = std::pair<Tensor, Tensor>;
static std::vector<pair_of> make_pair_vec(const std::vector<Tensor>& vals) {
  TORCH_CHECK(vals.size() % 2 == 0, "Odd number of params or hiddens given to a bidirectional RNN");
  std::vector<pair_of> result;
  result.reserve(vals.size() / 2);
  for (size_t i = 0; i < vals.size(); i += 2) {
    result.emplace_back(vals[i], vals[i + 1]);
  }
  return result;
}

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

tuple<Tensor, Tensor> gru_single_layer_bidirec_npu(
    const Tensor& input,
    pair_of& hx,
    BidirectCellParams params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  Tensor fw_weight_input = params.first.w_ih.t();
  Tensor fw_weight_hidden = params.first.w_hh.t();
  Tensor rev_weight_input = params.second.w_ih.t();
  Tensor rev_weight_hidden = params.second.w_hh.t();
  Tensor fw_bias_input;
  Tensor fw_bias_hidden;
  Tensor rev_bias_input;
  Tensor rev_bias_hidden;
  if (has_biases) {
    fw_bias_input = params.first.b_ih.to(input.dtype());
    fw_bias_hidden = params.first.b_hh.to(input.dtype());
    rev_bias_input = params.second.b_ih.to(input.dtype());
    rev_bias_hidden = params.second.b_hh.to(input.dtype());
  } else {
    fw_bias_input = OpPreparation::ApplyTensorWithFormat(fw_weight_input.size(1), input.options(), ACL_FORMAT_FRACTAL_NZ).mul(0);
    fw_bias_hidden = OpPreparation::ApplyTensorWithFormat(fw_weight_hidden.size(1), input.options(), ACL_FORMAT_FRACTAL_NZ).mul(0);
    rev_bias_input = OpPreparation::ApplyTensorWithFormat(rev_weight_input.size(1), input.options(), ACL_FORMAT_FRACTAL_NZ).mul(0);
    rev_bias_hidden = OpPreparation::ApplyTensorWithFormat(rev_weight_hidden.size(1), input.options(), ACL_FORMAT_FRACTAL_NZ).mul(0);
  }
  Tensor seq_length = OpPreparation::ApplyTensorWithFormat({}, input.options(), ACL_FORMAT_ND);
  auto results = at::npu_gru(
      input,
      hx.first,
      fw_weight_input,
      fw_weight_hidden,
      fw_bias_input,
      fw_bias_hidden,
      seq_length,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first);
  int64_t numStep = input.size(0);
  Tensor fw_output_hy = at::unsqueeze(std::get<1>(results)[numStep - 1], 0);
  Tensor fw_output = std::get<0>(results);
  auto rev_inputs = at::flip(input, {0}); // reverse input;
  auto rev_results = at::npu_gru(
      rev_inputs,
      hx.second,
      rev_weight_input,
      rev_weight_hidden,
      rev_bias_input,
      rev_bias_hidden,
      seq_length,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first);
  Tensor rev_output_hy = at::unsqueeze(std::get<1>(rev_results)[numStep - 1], 0);
  Tensor rev_output = at::flip(std::get<0>(rev_results),{0});
  return std::make_tuple(at::cat({fw_output, rev_output}, -1),
                         at::cat({fw_output_hy, rev_output_hy}));
}

tuple<Tensor, Tensor> gru_single_layer_direc_npu(
    const Tensor& input,
    const Tensor& hx,
    CellParams params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  // get weight  fp16
  Tensor weight_input = params.w_ih.t();
  Tensor weight_hidden = params.w_hh.t();

  // get bias  fp16 / fp32
  Tensor bias_input;
  Tensor bias_hidden;
  if (has_biases) {
    bias_input = params.b_ih.to(input.dtype());
    bias_hidden = params.b_hh.to(input.dtype());
  } else {
    bias_input = OpPreparation::ApplyTensorWithFormat(weight_input.size(1), input.options(), ACL_FORMAT_FRACTAL_NZ).mul(0);
    bias_hidden = OpPreparation::ApplyTensorWithFormat(weight_hidden.size(1), input.options(), ACL_FORMAT_FRACTAL_NZ).mul(0);
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

tuple<Tensor, Tensor> apply_layer_stack(
    const Tensor& input,
    std::vector<pair_of> hx,
    std::vector<pair_of> params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  auto layer_input = input;
  auto hidden_it = hx.begin();
  auto params_size = params.size();

  std::vector<BidirectCellParams> weights;
  std::vector<pair_of>::iterator params_it = params.begin();
  if (has_biases) {
    for (int64_t i = 0; i < params_size; i = i + 4){
      weights.emplace_back(CellParams((*(params_it+i)).first, (*(params_it+i)).second,
                                      (*(params_it+i+1)).first, (*(params_it+i+1)).second),
                           CellParams((*(params_it+i+2)).first, (*(params_it+i+2)).second,
                                      (*(params_it+i+3)).first, (*(params_it+i+3)).second));
    }
  } else {
    for (int64_t i = 0; i < params_size; i = i + 2){
      weights.emplace_back(CellParams((*(params_it+i)).first, (*(params_it+i)).second),
                           CellParams((*(params_it+i+1)).first, (*(params_it+i+1)).second));
    }
  }
  auto weights_it = weights.begin();
  std::vector<Tensor> final_hiddens;
  for (int64_t l = 0; l < num_layers; ++l) {
      auto layer_output = gru_single_layer_bidirec_npu(
          layer_input,
          *(hidden_it++),
          *(weights_it++),
          has_biases,
          num_layers,
          dropout,
          train,
          bidirectional,
          batch_first);
      final_hiddens.push_back(std::move(std::get<1>(layer_output)));
      layer_input = std::get<0>(layer_output);
    }
  return std::make_tuple(layer_input, at::cat(final_hiddens, 0));
}

tuple<Tensor, Tensor> apply_layer_stack(
    const Tensor& input,
    std::vector<Tensor>& hx,
    std::vector<Tensor>& params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  auto layer_input = input;
  auto hidden_it = hx.begin();

  auto params_size = params.size();
  std::vector<CellParams> weights;
  std::vector<Tensor>::iterator params_it = params.begin();
  if (has_biases) {
    for (int64_t i = 0; i < params_size; i = i + 4){
      weights.emplace_back(CellParams(*(params_it+i), *(params_it+i+1),
                                      *(params_it+i+2), *(params_it+i+3)));
    }
  } else {
    for (int64_t i = 0; i < params_size; i = i + 2){
      weights.emplace_back(CellParams(*(params_it+i), *(params_it+i+1)));
    }
  }
  auto weights_it = weights.begin();
  std::vector<Tensor> final_hiddens;

  for (int64_t l = 0; l < num_layers; ++l) {
    auto layer_output = gru_single_layer_direc_npu(
        layer_input,
        *(hidden_it++),
        *(weights_it++),
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first);
    final_hiddens.push_back(std::move(std::get<1>(layer_output)));
    layer_input = std::get<0>(layer_output);
  }
  auto hidden_state = at::cat(final_hiddens, 0);
  return std::make_tuple(layer_input, hidden_state);
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
  auto layer_hx = hx.unbind(0);
  int64_t total_layers = layer_hx.size();
  std::vector<Tensor> hiddens;
  for (int64_t i = 0; i < total_layers; ++i) {
    hiddens.emplace_back(std::move(layer_hx[i]));
  }
  std::vector<Tensor> paramsVec;
  for (int64_t i = 0; i < params.size(); ++i) {
    paramsVec.emplace_back(std::move(params[i]));
  }
  tuple<Tensor, Tensor> result;
  if (bidirectional) {
      result = apply_layer_stack(
          input,
          make_pair_vec(hiddens),
          make_pair_vec(paramsVec),
          has_biases,
          num_layers,
          dropout,
          train,
          bidirectional,
          batch_first);
  } else {
      result = apply_layer_stack(
          input,
          hiddens,
          paramsVec,
          has_biases,
          num_layers,
          dropout,
          train,
          bidirectional,
          batch_first);
  }
  return result;
}

} // namespace native
} // namespace at