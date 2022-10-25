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

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

struct CellParams {
  CellParams(const at::Tensor& _w_ih, const at::Tensor& _w_hh)
    : w_ih(_w_ih), w_hh(_w_hh), b_ih({}), b_hh({}) {};
  CellParams(const at::Tensor& _w_ih, const at::Tensor& _w_hh, const at::Tensor& _b_ih, const at::Tensor& _b_hh)
    : w_ih(_w_ih), w_hh(_w_hh), b_ih(_b_ih), b_hh(_b_hh) {};
  const at::Tensor& w_ih;
  const at::Tensor& w_hh;
  const at::Tensor& b_ih; /* optional */
  const at::Tensor& b_hh; /* optional */
};

using BidirectCellParams = std::pair<CellParams, CellParams>;
using pair_of = std::pair<at::Tensor, at::Tensor>;
static std::vector<pair_of> make_pair_vec(const std::vector<at::Tensor>& vals) {
  TORCH_CHECK(vals.size() % 2 == 0, "Odd number of params or hiddens given to a bidirectional RNN");
  std::vector<pair_of> result;
  result.reserve(vals.size() / 2);
  for (size_t i = 0; i < vals.size(); i += 2) {
    result.emplace_back(vals[i], vals[i + 1]);
  }
  return result;
}

std::vector<at::Tensor> gru_npu(
    const at::Tensor& input,
    const at::Tensor& hx,
    const at::Tensor& weight_input,
    const at::Tensor& weight_hidden,
    const at::Tensor& bias_input,
    const at::Tensor& bias_hidden,
    const at::Tensor& seq_length,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  int64_t numStep = input.size(0);
  int64_t batchSize = input.size(1);
  int64_t hiddenSize = bias_input.size(0) / 3;
  c10::SmallVector<int64_t, SIZE> outputSize = {numStep, batchSize, hiddenSize};

  at::Tensor output_y = OpPreparation::ApplyTensor(bias_input, outputSize);
  at::Tensor output_h = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      bias_input.options(),
      ACL_FORMAT_ND);
  at::Tensor output_updata = OpPreparation::ApplyTensor(bias_input, outputSize);
  at::Tensor output_reset = OpPreparation::ApplyTensor(bias_input, outputSize);
  at::Tensor output_new = OpPreparation::ApplyTensor(bias_input, outputSize);
  at::Tensor hidden_new = OpPreparation::ApplyTensor(bias_input, outputSize);

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
  tensor_list results = {output_y, output_h, output_updata, output_reset, output_new, hidden_new};
  return results;
}

std::vector<at::Tensor> NPUNativeFunctions::npu_gru_backward(
    const c10::optional<at::Tensor>& grady_opt,
    const c10::optional<at::Tensor>& gradh_opt,
    const at::Tensor& input,
    const at::Tensor& weight_input,
    const at::Tensor& weight_hidden,
    const at::Tensor& bias_input,
    const at::Tensor& bias_hidden,
    const at::Tensor& seq_length,
    const at::Tensor& init_h,
    const at::Tensor& output_y,
    const at::Tensor& output_h,
    const at::Tensor& output_updata,
    const at::Tensor& output_reset,
    const at::Tensor& output_new,
    const at::Tensor& hidden_new) {
  const at::Tensor& grady = c10::value_or_else(grady_opt, [] {return at::Tensor();});
  const at::Tensor& gradh = c10::value_or_else(gradh_opt, [] {return at::Tensor();});
  at::Tensor inh = at::squeeze(init_h, 0);
  auto grad_y =
      grady.defined() ? grady : OpPreparation::ApplyTensor(output_y).mul(0);
  auto grad_h =
      gradh.defined() ? gradh[input.size(0)-1] : OpPreparation::ApplyTensor(output_h, inh.sizes()).mul(0);

  at::Tensor mask = at::zeros({}, input.options().dtype(at::kByte)); // uint8
  at::Tensor seq_lengths = at::zeros({}, input.options());

  int64_t npu_format = ACL_FORMAT_ND;

  at::Tensor grad_w_input = OpPreparation::ApplyTensorWithFormat(weight_input.sizes(), input.options(), npu_format);
  at::Tensor grad_w_hidden = OpPreparation::ApplyTensorWithFormat(weight_hidden.sizes(), input.options(), npu_format);
  at::Tensor grad_x = OpPreparation::ApplyTensorWithFormat(input.sizes(), input.options(), npu_format);
  at::Tensor grad_b_input = OpPreparation::ApplyTensorWithFormat(bias_input.sizes(), input.options(), npu_format);
  at::Tensor grad_b_hidden = OpPreparation::ApplyTensorWithFormat(bias_hidden.sizes(), input.options(), npu_format);
  at::Tensor grad_h_prev = OpPreparation::ApplyTensorWithFormat(init_h.sizes(), input.options(), npu_format);

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
  tensor_list results = {grad_x, grad_h_prev, grad_w_input, grad_w_hidden, grad_b_input, grad_b_hidden};
  return results;
}

class NPUGruFunction : public torch::autograd::Function<NPUGruFunction> {
public:
  static tensor_list forward(AutogradContext *ctx,
    const at::Tensor& input,
    const at::Tensor& hx,
    const at::Tensor& weight_input,
    const at::Tensor& weight_hidden,
    const at::Tensor& bias_input,
    const at::Tensor& bias_hidden,
    const at::Tensor& seq_length,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
    auto result = gru_npu(input, hx, weight_input, weight_hidden,
        bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first);
    auto result0 = result[0];
    auto result1 = result[1];
    auto result2 = result[2];
    auto result3 = result[3];
    auto result4 = result[4];
    auto result5 = result[5];

    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({weight_input, 
        weight_hidden, 
        input, 
        bias_input, 
        bias_hidden, 
        hx,
        result0,
        result1,
        result2,
        result3,
        result4,
        result5,
        seq_length
        });
    return result;
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto weight_input = saved[0];
    auto weight_hidden = saved[1];
    auto input = saved[2];
    auto bias_input = saved[3];
    auto bias_hidden = saved[4];
    auto hx = saved[5];
    auto result0 = saved[6];
    auto result1 = saved[7];
    auto result2 = saved[8];
    auto result3 = saved[9];
    auto result4 = saved[10];
    auto result5 = saved[11];
    auto seq_length = saved[12];
    
    tensor_list result = NPUNativeFunctions::npu_gru_backward(
        grad_outputs[0],
        grad_outputs[1],
        input,
        weight_input,
        weight_hidden,
        bias_input,
        bias_hidden,
        seq_length,
        hx,
        result0,
        result1,
        result2,
        result3,
        result4,
        result5);

    tensor_list output = {
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        result[5],
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
    return output;
  }
};
std::vector<at::Tensor> NPUNativeFunctions::npu_gru(
    const at::Tensor& input,
    const at::Tensor& hx,
    const at::Tensor& weight_input,
    const at::Tensor& weight_hidden,
    const at::Tensor& bias_input,
    const at::Tensor& bias_hidden,
    const at::Tensor& seq_length,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  return NPUGruFunction::apply(input, hx, weight_input, weight_hidden,
        bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

tuple<at::Tensor, at::Tensor> gru_single_layer_bidirec_npu(
    const at::Tensor& input,
    pair_of& hx,
    BidirectCellParams params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  at::Tensor fw_weight_input = params.first.w_ih.t();
  at::Tensor fw_weight_hidden = params.first.w_hh.t();
  at::Tensor rev_weight_input = params.second.w_ih.t();
  at::Tensor rev_weight_hidden = params.second.w_hh.t();
  at::Tensor fw_bias_input;
  at::Tensor fw_bias_hidden;
  at::Tensor rev_bias_input;
  at::Tensor rev_bias_hidden;
  if (has_biases) {
    fw_bias_input = params.first.b_ih.to(input.dtype());
    fw_bias_hidden = params.first.b_hh.to(input.dtype());
    rev_bias_input = params.second.b_ih.to(input.dtype());
    rev_bias_hidden = params.second.b_hh.to(input.dtype());
  } else {
    fw_bias_input = OpPreparation::ApplyTensor(input, fw_weight_input.size(1)).mul(0);
    fw_bias_hidden = OpPreparation::ApplyTensor(input, fw_weight_hidden.size(1)).mul(0);
    rev_bias_input = OpPreparation::ApplyTensor(input, rev_weight_input.size(1)).mul(0);
    rev_bias_hidden = OpPreparation::ApplyTensor(input, rev_weight_hidden.size(1)).mul(0);
  }
  at::Tensor seq_length = OpPreparation::ApplyTensorWithFormat({}, input.options(), ACL_FORMAT_ND);
  auto results = NPUNativeFunctions::npu_gru(
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
  at::Tensor fw_output_hy = at::unsqueeze(results[1][numStep - 1], 0);
  at::Tensor fw_output = results[0];
  auto rev_inputs = at::flip(input, {0}); // reverse input;
  auto rev_results = NPUNativeFunctions::npu_gru(
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
  at::Tensor rev_output_hy = at::unsqueeze(rev_results[1][numStep - 1], 0);
  at::Tensor rev_output = at::flip(rev_results[0],{0});
  return std::make_tuple(at::cat({fw_output, rev_output}, -1),
                         at::cat({fw_output_hy, rev_output_hy}));
}

tuple<at::Tensor, at::Tensor> gru_single_layer_direc_npu(
    const at::Tensor& input,
    const at::Tensor& hx,
    CellParams params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  // get weight  fp16
  at::Tensor weight_input = params.w_ih.t();
  at::Tensor weight_hidden = params.w_hh.t();

  // get bias  fp16 / fp32
  at::Tensor bias_input;
  at::Tensor bias_hidden;
  if (has_biases) {
    bias_input = params.b_ih.to(input.dtype());
    bias_hidden = params.b_hh.to(input.dtype());
  } else {
    bias_input = OpPreparation::ApplyTensor(input, weight_input.size(1)).mul(0);
    bias_hidden = OpPreparation::ApplyTensor(input, weight_hidden.size(1)).mul(0);
  }

  at::Tensor seq_length = OpPreparation::ApplyTensorWithFormat({}, input.options(), ACL_FORMAT_ND);

  auto results = NPUNativeFunctions::npu_gru(
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
  at::Tensor output_hy = at::unsqueeze(results[1][numStep - 1], 0);
  return std::tuple<at::Tensor, at::Tensor>(results[0], output_hy);
}

tuple<at::Tensor, at::Tensor> apply_layer_stack(
    const at::Tensor& input,
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
  std::vector<at::Tensor> final_hiddens;
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

tuple<at::Tensor, at::Tensor> apply_layer_stack(
    const at::Tensor& input,
    std::vector<at::Tensor>& hx,
    std::vector<at::Tensor>& params,
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
  std::vector<at::Tensor>::iterator params_it = params.begin();
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
  std::vector<at::Tensor> final_hiddens;

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

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::gru(
    const at::Tensor& input_,
    const at::Tensor& hx,
    at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) {
  // The operator of DynamicGRU only supports the T axis as the first axis.
  auto input = batch_first ? input_.transpose(0, 1) : input_;

  auto layer_hx = hx.unbind(0);
  int64_t total_layers = layer_hx.size();
  std::vector<at::Tensor> hiddens;
  for (int64_t i = 0; i < total_layers; ++i) {
    hiddens.emplace_back(std::move(layer_hx[i]));
  }
  std::vector<at::Tensor> paramsVec;
  for (int64_t i = 0; i < params.size(); ++i) {
    paramsVec.emplace_back(std::move(params[i]));
  }
  tuple<at::Tensor, at::Tensor> result;
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
  std::get<0>(result) = batch_first ? std::get<0>(result).transpose(0, 1) : std::get<0>(result);
  return result;
}

} // namespace native
} // namespace at_npu