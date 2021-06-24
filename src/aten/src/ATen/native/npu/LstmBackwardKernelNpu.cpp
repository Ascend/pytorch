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

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&> lstm_backward_out_npu(
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

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> lstm_backward_npu(
    const Tensor& grady, 
    const Tensor& gradh, 
    const Tensor& gradc, 
    const Tensor& input, 
    const Tensor& weight,
    const Tensor& bias, 
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

  Tensor inh = at::squeeze(init_h, 0);
  Tensor inc = at::squeeze(init_c, 0);

  Tensor grad_input = OpPreparation::ApplyTensor(input); 
  Tensor grad_weight = OpPreparation::ApplyTensor(weight);
  Tensor grad_bias = OpPreparation::ApplyTensor(bias);
  Tensor grad_ht = OpPreparation::ApplyTensor(init_h);
  Tensor grad_ct = OpPreparation::ApplyTensor(init_c);
  
  auto grad_y = grady.defined() ? grady : at::zeros(y.sizes(), y.options());
  auto grad_h = gradh.defined() ? gradh[input.size(0)-1] : at::zeros(inh.sizes(), h.options());
  auto grad_c = gradc.defined() ? gradc[input.size(0)-1] : at::zeros(inc.sizes(), c.options());

  lstm_backward_out_npu(grad_weight, grad_bias, grad_input, grad_ht, grad_ct, input, weight,
                        bias, inh, inc, grad_y, grad_h, grad_c, y, h, c, i, j, f, o, tanhc);

  return std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> {grad_input, grad_weight, grad_bias, grad_ht, grad_ct};
}

} // namespace native 
} // namespace at 