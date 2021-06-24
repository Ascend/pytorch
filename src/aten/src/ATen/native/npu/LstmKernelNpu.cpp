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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> lstm_npu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& h,
    const Tensor& c,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first) { 
  // calculate the output size
  int64_t numStep = input.size(0);
  int64_t batchSize = input.size(1);
  int64_t hiddenSize = bias.size(0) / 4;

  SmallVector<int64_t, SIZE> outputSize = {numStep, batchSize, hiddenSize};

  // construct the output tensor of the NPU
  Tensor yOutput = OpPreparation::ApplyTensor(input, outputSize);
  Tensor hOutput = OpPreparation::ApplyTensor(input, outputSize);
  Tensor cOutput = OpPreparation::ApplyTensor(input, outputSize);
  Tensor iOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  Tensor jOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  Tensor fOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  Tensor oOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  Tensor tanhc = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ); 
  
  OpCommand cmd;
  cmd.Name("DynamicRNN")
      .Input(input)
      .Input(weight)
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
      .Attr("is_training", train)
      .Run();

  return std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(
      yOutput, hOutput, cOutput, iOutput, jOutput, fOutput, oOutput, tanhc);
}

tuple<Tensor, Tensor, Tensor> lstm_npu(
    const Tensor& input,
    TensorList hx,
    TensorList params,
    bool hasBiases,
    int64_t numLayers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batchFirst) { 
  int64_t numStep = input.size(0);
  
  // get weight
  Tensor ihWeight = params[0];
  Tensor hhWeight = params[1];
	
  Tensor weight = at::cat({ihWeight, hhWeight}, 1).t().to(input.dtype());
  
  // get bias
  Tensor bias = at::zeros(weight.size(1), weight.options());
  if (hasBiases) {
    bias = at::add(params[2], params[3]).to(input.dtype());
  }
  
  // get init_h, init_c 
  Tensor h = hx[0];
  Tensor c = hx[1];
  if(numLayers == 2)
  {
    h = hx[0].slice(0, 0, 1);
    c = hx[1].slice(0, 0, 1);
  }

  auto results = at::npu_lstm(
    input, weight, bias, h, c, hasBiases, numLayers, dropout, train, bidirectional, batchFirst);

  // get the last dimension of the T-axis	
  Tensor thOutput = at::unsqueeze(std::get<1>(results)[numStep-1], 0);
  Tensor tcOutput = at::unsqueeze(std::get<2>(results)[numStep-1], 0);
  
  //double layer LSTM
  if (numLayers == 2) {
    Tensor weight2Layer;
    Tensor bias2Layer;
    Tensor h2layer = hx[0].slice(0, 1, 2);
    Tensor c2layer = hx[1].slice(0, 1, 2);
    if (hasBiases) {
      weight2Layer = at::cat({params[4], params[5]}, 1).t().to(input.dtype());
      bias2Layer = at::add(params[6], params[7]).to(input.dtype());
    } else {
      weight2Layer = at::cat({params[2], params[3]}, 1).t().to(input.dtype());
      bias2Layer = at::zeros(weight2Layer.size(1), weight2Layer.options());
    }
    
    //output of first layer as input of second layer
    Tensor input2Layer = std::get<0>(results);
    
    //caculate output of second layer
    auto results2Layer = at::npu_lstm(input2Layer, weight2Layer, bias2Layer, h2layer, c2layer, 
    hasBiases, numLayers, dropout, train, bidirectional, batchFirst);
    Tensor thOutput2Layer = at::unsqueeze(std::get<1>(results2Layer)[numStep-1], 0);
    Tensor tcOutput2Layer = at::unsqueeze(std::get<2>(results2Layer)[numStep-1], 0);
    Tensor th = at::cat({thOutput, thOutput2Layer}, 0);
    Tensor tc = at::cat({tcOutput, tcOutput2Layer}, 0);

    return std::tie(std::get<0>(results2Layer), th, tc);
  }
  
  return std::tie(std::get<0>(results), thOutput, tcOutput);
}

std::tuple<Tensor, Tensor, Tensor> lstm_npu(
  const Tensor& data, const Tensor& batchSizes, TensorList hx,
  TensorList params, bool hasBiases,
  int64_t numLayers, double dropoutP, bool train, bool bidirectional) {
  //length of T axis
  int64_t t_size = batchSizes.numel();
  
  //T * B **
  Tensor input = data.reshape({t_size, data.size(0)/t_size, data.size(1)});

  // batch_first is false
  bool batchFirst = false;

  // get init_h, init_c 
  Tensor h = hx[0];
  Tensor c = hx[1];

  int64_t maxLen = input.size(0);
  std::vector<at::Tensor> outputs;
  std::vector<at::Tensor> hxPrev = {h, c}; 

  //caculate lengths, but input expected to be sorted
  std::vector<int64_t> lens;
  for (int64_t i = 0; i < input.size(1); ++i){
    auto batchSizesTemp = at::sub(batchSizes , i);
    auto batchSizesBool = at::gt(batchSizesTemp, 0); 
    auto batchSizesInt = batchSizesBool.to(ScalarType::Int);
    auto coutLen = at::sum(batchSizesInt, ScalarType::Int);
    int64_t len = coutLen.item().toInt();
    lens.emplace_back(len);
  }
  Tensor length = CalcuOpUtil::copy_tensor_host_to_device(
    from_blob(lens.data(), {lens.size()}, at::kLong));

  //Slice by T axis
  for (int64_t i = 0; i < maxLen; ++i) {
    Tensor step = input.slice(0, i, i + 1).contiguous().reshape({1, input.size(1), input.size(2)});
    
    //calculate output of each times
    auto results = lstm_npu(step, hxPrev, params, hasBiases, numLayers, dropoutP, train, bidirectional, batchFirst);
    
    //get previous result
    Tensor outputTemp = std::get<0>(results);
    std::vector<at::Tensor> hxCurr = {std::get<1>(results), std::get<2>(results)};

    //cacl mask
    Tensor maskTemp = at::gt(length, i);
    Tensor mask = maskTemp.reshape({1, input.size(1), 1});

    //calculate real output of each times
    Tensor maskNeg = at::logical_not(mask);
    Tensor output = at::mul(outputTemp, mask);

    //updata hx
    h = at::mul(mask, hxCurr[0]) + at::mul(maskNeg, hxPrev[0]);
    c = at::mul(mask, hxCurr[1]) + at::mul(maskNeg, hxPrev[1]);    
    hxPrev = {h, c};

    outputs.push_back(output);
  }  
  Tensor result = at::cat(outputs, 0);
  
  return std::tie(result, h, c);    
}

} // namespace native
} // namespace at