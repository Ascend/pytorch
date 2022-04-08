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

#include "c10/npu/OptionsManager.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, Tensor&> lstm_cell_backward_nocheck_npu(
    Tensor& dx, 
    Tensor& dw_x, 
    Tensor& dw_h, 
    Tensor& db, 
    Tensor& dht, 
    Tensor& dct,
    const Tensor& dy, 
    const Tensor& dh, 
    const Tensor& dc,
    const Tensor& x, 
    const Tensor& w_x, 
    const Tensor& w_h, 
    const Tensor& init_h, 
    const Tensor& init_c, 
    const Tensor& y, 
    const Tensor& h, 
    const Tensor& c, 
    const Tensor& i,
    const Tensor& j, 
    const Tensor& f, 
    const Tensor& o, 
    const Tensor& tanhc) {
  Tensor seq_length = at::zeros({}, x.options());
  Tensor mask = at::zeros({}, x.options().dtype(kByte));
  Tensor wci = at::zeros({}, x.options());
  Tensor wcf = at::zeros({}, x.options());
  Tensor wco = at::zeros({}, x.options());
  OpCommand cmd;
    cmd.Name("DynamicRNNV2Grad")
        .Input(x)
        .Input(w_x)
        .Input(w_h)
        .Input(y)
        .Input(init_h)
        .Input(init_c)
        .Input(h)
        .Input(c)
        .Input(dy)
        .Input(dh)
        .Input(dc)
        .Input(i)
        .Input(j)
        .Input(f)
        .Input(o)
        .Input(tanhc)
        .Input(seq_length)
        .Input(wci)
        .Input(wcf)
        .Input(wco)
        .Input(mask)
        .Output(dw_x)
        .Output(dw_h)
        .Output(db)
        .Output(dx)
        .Output(dht)
        .Output(dct)
        .Attr("cell_type", (string)"LSTM")
        .Attr("direction", (string)"UNIDIRECTIONAL")
        .Attr("cell_depth", (int64_t)1)
        .Attr("use_peephole", (bool)false)
        .Attr("keep_prob", (float)1.0)
        .Attr("cell_clip", (float)-1.0)
        .Attr("num_proj", (int64_t)0)
        .Attr("time_major", (bool)true)
        .Attr("activation", (string)"tanh")
        .Attr("recurrent_activation", (string)"sigmoid")
        .Attr("gate_order", (string)"ifjo")
        .Attr("stateful", (bool)false)
        .Attr("merge_mode", (string)"concat")
        .Run();
  return std::tie(dx, dw_x, dw_h, db, dht, dct);
}

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&> lstm_backward_nocheck_npu(
    Tensor& dw, 
    Tensor& db, 
    Tensor& dx, 
    Tensor& dht, 
    Tensor& dct,
    const Tensor& x, 
    const Tensor& w, 
    const Tensor& b, 
    const Tensor& init_h, 
    const Tensor& init_c, 
    const Tensor& dy, 
    const Tensor& dh, 
    const Tensor& dc,
    const Tensor& y, 
    const Tensor& h, 
    const Tensor& c, 
    const Tensor& i,
    const Tensor& j, 
    const Tensor& f, 
    const Tensor& o, 
    const Tensor& tanhc) {
  
  Tensor seq_length = at::zeros({}, x.options());
  Tensor mask = at::zeros({}, x.options().dtype(kByte));
  Tensor wci = at::zeros({}, x.options());
  Tensor wcf = at::zeros({}, x.options());
  Tensor wco = at::zeros({}, x.options());

  OpCommand cmd;
    cmd.Name("DynamicRNNGrad")
        .Input(x)
        .Input(w)
        .Input(b)
        .Input(y)
        .Input(init_h)
        .Input(init_c)
        .Input(h)
        .Input(c)
        .Input(dy)
        .Input(dh)
        .Input(dc)
        .Input(i)
        .Input(j)
        .Input(f)
        .Input(o)
        .Input(tanhc)
        .Input(seq_length)
        .Input(mask)
        .Input(wci)
        .Input(wcf)
        .Input(wco)
        .Output(dw)
        .Output(db)
        .Output(dx)
        .Output(dht)
        .Output(dct)
        .Attr("cell_type", "LSTM")
        .Attr("direction", "UNIDIRECTIONAL")
        .Attr("cell_depth", (int64_t)0)
        .Attr("use_peephole", (bool)false)
        .Attr("keep_prob", (float)-1.0)
        .Attr("cell_clip", (float)-1.0)
        .Attr("num_proj", (int64_t)0)
        .Attr("time_major", (bool)true)
        .Attr("forget_bias", (float)0.0)
        .Run();

  return std::tuple< Tensor&, Tensor&, Tensor&, Tensor&, Tensor&> {dx, dw, db, dht, dct};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> lstm_cell_backward_npu(
    const Tensor& grady,
    const Tensor& gradh,
    const Tensor& gradc,
    const Tensor& _input,
    const Tensor& w_ix,
    const Tensor& w_ih,
    const Tensor& _h,
    const Tensor& _c,
    const Tensor& y_,
    const Tensor& h_,
    const Tensor& c_,
    const Tensor& i_,
    const Tensor& j_,
    const Tensor& f_,
    const Tensor& o_,
    const Tensor& tanhc_) { 
  int64_t batchSzie = y_.size(1);
  int64_t hiddenSize = y_.size(2);

  if (hiddenSize % 16 == 0){
    Tensor input = _input.reshape({batchSzie, _input.size(1)});
    Tensor inh = _h.reshape({batchSzie, hiddenSize});
    Tensor inc = _c.reshape({batchSzie, hiddenSize});
    Tensor w_x = w_ix.reshape({_input.size(1), 4 * hiddenSize});
    Tensor w_h = w_ih.reshape({hiddenSize, 4 * hiddenSize});
    Tensor y = y_.reshape({batchSzie, hiddenSize});
    Tensor h = h_.reshape({batchSzie, hiddenSize});
    Tensor c = c_.reshape({batchSzie, hiddenSize});
    Tensor i = i_.reshape({batchSzie, hiddenSize});
    Tensor j = j_.reshape({batchSzie, hiddenSize});
    Tensor f = f_.reshape({batchSzie, hiddenSize});
    Tensor o = o_.reshape({batchSzie, hiddenSize});
    Tensor tanhc = tanhc_.reshape({batchSzie, hiddenSize});

    SmallVector<int64_t, 8> outputSize = {4 * hiddenSize};
    Tensor grad_input = OpPreparation::ApplyTensor(input);
    Tensor grad_wx = OpPreparation::ApplyTensor(w_x);
    Tensor grad_wh = OpPreparation::ApplyTensor(w_h);
    Tensor grad_bias = OpPreparation::ApplyTensor(input, outputSize);
    Tensor grad_ht = OpPreparation::ApplyTensor(inh);
    Tensor grad_ct = OpPreparation::ApplyTensor(inc);
 
    auto grad_y = grady.defined() ? grady : at::zeros(y.sizes(), y.options());
    auto grad_h = gradh.defined() ? gradh : at::zeros(inh.sizes(), h.options());
    auto grad_c = gradc.defined() ? gradc : at::zeros(inc.sizes(), c.options());

    lstm_cell_backward_nocheck_npu(grad_input, grad_wx, grad_wh, grad_bias, grad_ht, grad_ct, grad_y, grad_h, grad_c, 
                                   input, w_x, w_h, inh, inc, y, h, c, i, j, f, o, tanhc);
    return std::tie(grad_input, grad_wx, grad_wh, grad_bias, grad_ht, grad_ct);
  } else {
    Tensor weight = at::cat({w_ix, w_ih}, 0).to(_input.dtype());
    Tensor weight1 = at::npu_slice(weight, {0, 0}, {weight.size(0), hiddenSize});
    Tensor weight2 = at::npu_slice(weight, {0, hiddenSize}, {weight.size(0), hiddenSize});
    Tensor weight3 = at::npu_slice(weight, {0, 2 * hiddenSize}, {weight.size(0), hiddenSize});
    Tensor weight4 = at::npu_slice(weight, {0, 3 * hiddenSize}, {weight.size(0), hiddenSize});
    weight = at::cat({weight1, weight3, weight2, weight4}, 1);

    Tensor bias = at::zeros({4 * hiddenSize}, _c.options());
    Tensor input = _input.reshape({1, batchSzie, _input.size(1)});
    Tensor inh = _h.reshape({batchSzie, hiddenSize});
    Tensor inc = _c.reshape({batchSzie, hiddenSize});
    Tensor w_x = w_ix.reshape({_input.size(1), 4 * hiddenSize});
    Tensor w_h = w_ih.reshape({hiddenSize, 4 * hiddenSize});
    Tensor y = y_.reshape({1, batchSzie, hiddenSize});
    Tensor h = h_.reshape({1, batchSzie, hiddenSize});
    Tensor c = c_.reshape({1, batchSzie, hiddenSize});
    Tensor i = i_.reshape({1, batchSzie, hiddenSize});
    Tensor j = j_.reshape({1, batchSzie, hiddenSize});
    Tensor f = f_.reshape({1, batchSzie, hiddenSize});
    Tensor o = o_.reshape({1, batchSzie, hiddenSize});
    Tensor tanhc = tanhc_.reshape({1, batchSzie, hiddenSize});

    Tensor grad_input = OpPreparation::ApplyTensor(input); 
    Tensor grad_weight = OpPreparation::ApplyTensor(weight);
    Tensor grad_bias = OpPreparation::ApplyTensor(bias);
    Tensor grad_ht = OpPreparation::ApplyTensor(h);
    Tensor grad_ct = OpPreparation::ApplyTensor(c);
    auto grad_y = grady.defined() ? grady : at::zeros(y.sizes(), y.options());
    auto grad_h = gradh.defined() ? gradh : at::zeros(inh.sizes(), h.options());
    auto grad_c = gradc.defined() ? gradc : at::zeros(inc.sizes(), c.options());

    lstm_backward_nocheck_npu(grad_weight, grad_bias, grad_input, grad_ht, grad_ct, input, weight,
                              bias, inh, inc, grad_y, grad_h, grad_c, y, h, c, i, j, f, o, tanhc);

    Tensor weight_opt1 = at::npu_slice(grad_weight, {0, 0}, {_input.size(1), 4 * hiddenSize});
    Tensor weight_opt2 = at::npu_slice(grad_weight, {_input.size(1), 0}, {hiddenSize, 4 * hiddenSize});
    grad_input = grad_input.reshape({batchSzie, _input.size(1)});
    grad_ht = grad_ht.reshape({batchSzie, hiddenSize});
    grad_ct = grad_ct.reshape({batchSzie, hiddenSize});
    return std::tie(grad_input, weight_opt1, weight_opt2, grad_bias, grad_ht, grad_ct);  
  }
  
}
} // namespace native 
} // namespace at
