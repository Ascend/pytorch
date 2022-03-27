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

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

std::vector<at::Tensor> npu_lstm_npu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& seqMask,
    const at::Tensor& h,
    const at::Tensor& c,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first,
    bool flagSeq,
    bool flagDirection) { 
  // calculate the output size
  int64_t numStep = input.size(0);
  int64_t batchSize = input.size(1);
  int64_t hiddenSize = bias.size(0) / 4;

  c10::SmallVector<int64_t, SIZE> outputSize = {numStep, batchSize, hiddenSize};

  // construct the output tensor of the NPU
  at::Tensor yOutput = OpPreparation::ApplyTensor(input, outputSize);
  at::Tensor hOutput = OpPreparation::ApplyTensor(input, outputSize);
  at::Tensor cOutput = OpPreparation::ApplyTensor(input, outputSize);
  at::Tensor iOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor jOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor fOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor oOutput = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ);
  at::Tensor tanhc = OpPreparation::ApplyTensorWithFormat(input, outputSize, ACL_FORMAT_FRACTAL_NZ); 
 
  string direction = flagDirection? "REDIRECTIONAL" : "UNIDIRECTIONAL";
  OpCommand cmd;
  cmd.Name("DynamicRNN")
    .Input(input, "x")
    .Input(weight, "w")
    .Input(bias, "b");
     
  // if input is PackSequence, seqMask is not None,  Otherwise, it is None.   
  if (!flagSeq){
    cmd.Input();
  } else{
    cmd.Input(seqMask, "seq_length"); 
  }      
    cmd.Input(h, "init_h")
    .Input(c, "init_c")
    .Output(yOutput)
    .Output(hOutput)
    .Output(cOutput)
    .Output(iOutput)
    .Output(jOutput)
    .Output(fOutput)
    .Output(oOutput)
    .Output(tanhc)
    .Attr("cell_type", (string)"LSTM")
    .Attr("direction", direction)
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
  tensor_list results = {yOutput, hOutput, cOutput, iOutput, jOutput, fOutput, oOutput, tanhc};
  return results;
}

tuple<at::Tensor, at::Tensor> get_wb_single_layer_direc(
    const at::Tensor& input,
    at::TensorList params,
    bool hasBiases) {
  // get weight
  at::Tensor ihWeight = params[0];
  at::Tensor hhWeight = params[1];
	
  at::Tensor weight = at::cat({ihWeight, hhWeight}, 1).t().to(input.dtype());
  
  // get bias
  at::Tensor bias = at::zeros(weight.size(1), weight.options());
  if (hasBiases) {
    bias = at::add(params[2], params[3]).to(input.dtype());
  }
  return std::tie(weight, bias);
}

tuple<at::Tensor, at::Tensor> get_wb_double_layer_or_bidirec(
    const at::Tensor& input,
    at::TensorList params,
    bool hasBiases) {
  at::Tensor weight;
  at::Tensor bias; 
  if (hasBiases) {
    weight = at::cat({params[4], params[5]}, 1).t().to(input.dtype());
    bias = at::add(params[6], params[7]).to(input.dtype());
  } else {
    weight = at::cat({params[2], params[3]}, 1).t().to(input.dtype());
    bias = at::zeros(weight.size(1), weight.options());
  }
  return std::tie(weight, bias);
}

tuple<at::Tensor, at::Tensor, at::Tensor> lstm_single_layer_direc_npu(
    const at::Tensor& input,
    at::TensorList hx,
    at::TensorList params,
    bool hasBiases,
    int64_t numLayers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batchFirst,
    bool direction) {
  int64_t numStep = input.size(0);
  
  // get weight
  at::Tensor ihWeight = params[0];
  at::Tensor hhWeight = params[1];
	
  at::Tensor weight = at::cat({ihWeight, hhWeight}, 1).t().to(input.dtype());
  
  // get bias
  at::Tensor bias = at::zeros(weight.size(1), weight.options());
  if (hasBiases) {
    bias = at::add(params[2], params[3]).to(input.dtype());
  }
  
  // get init_h, init_c 
  at::Tensor h = hx[0];
  at::Tensor c = hx[1];
  
  at::Tensor seqMask = at::empty({0}, input.options());
  auto results = NPUNativeFunctions::npu_lstm(input, weight, bias, seqMask, h, c, hasBiases, numLayers, dropout, 
      train, bidirectional, batchFirst, false, direction);

  // get the last dimension of the T-axis	
  at::Tensor thOutput = at::unsqueeze(results[1][numStep-1], 0);
  at::Tensor tcOutput = at::unsqueeze(results[2][numStep-1], 0);

  return std::tie(results[0], thOutput, tcOutput);
}

tuple<at::Tensor, at::Tensor, at::Tensor> lstm_single_layer_bidirec_npu(
    const at::Tensor& input,
    at::TensorList hx,
    at::TensorList params,
    bool hasBiases,
    int64_t numLayers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batchFirst) {
  int64_t numStep = input.size(0);
  // get h and c of forward direction
  at::Tensor h = hx[0].slice(0, 0, 1);
  at::Tensor c = hx[1].slice(0, 0, 1);
  
  // caculate forward direction, direction of attr is UNIDIRECTIONAL(npu_lstm need add the attr of direction)
  auto resultsForward = lstm_single_layer_direc_npu(input, {h, c}, params, hasBiases, 
      numLayers, dropout, train, bidirectional, batchFirst, false); 

  // get w/ b/ h/ c of backward direction
  at::Tensor weightBack;
  at::Tensor biasBack;
  at::Tensor hBack = hx[0].slice(0, 1, 2);
  at::Tensor cBack = hx[1].slice(0, 1, 2);
  std::tie(weightBack, biasBack) = get_wb_double_layer_or_bidirec(input, params, hasBiases);

  at::Tensor seqMask = at::empty({0}, input.options());
  auto revInputs = at::flip(input, {0});

  // caculate backward direction, direction of attr is REDIRECTIONAL, 
  // but the inverse operator does not support the specified direction, 
  // it is necessary to flip the input and output at the adaptation layer.
  auto resultsBackward = NPUNativeFunctions::npu_lstm(revInputs, weightBack, biasBack, seqMask, hBack, cBack, 
      hasBiases, numLayers, dropout, train, bidirectional, batchFirst, false, false);
  
  // get the first dimension of the T-axis when caculate reverse direction
  at::Tensor revY = at::flip(resultsBackward[0],{0});
  at::Tensor th = at::flip(resultsBackward[1],{0});
  at::Tensor tc = at::flip(resultsBackward[2],{0});
  at::Tensor thOutput = at::unsqueeze(th[0], 0);
  at::Tensor tcOutput = at::unsqueeze(tc[0], 0);    

  at::Tensor y = at::cat({std::get<0>(resultsForward), revY}, 2); 
  at::Tensor hOut = at::cat({std::get<1>(resultsForward), thOutput}, 0);
  at::Tensor cOut = at::cat({std::get<2>(resultsForward), tcOutput}, 0);

  return std::tie(y, hOut, cOut);
}

tuple<at::Tensor, at::Tensor, at::Tensor> lstm_double_layer_direc_npu(
    const at::Tensor& input,
    at::TensorList hx,
    at::TensorList params,
    bool hasBiases,
    int64_t numLayers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batchFirst) {
  int64_t numStep = input.size(0);
  // get h and c of first layer
  at::Tensor h = hx[0].slice(0, 0, 1);
  at::Tensor c = hx[1].slice(0, 0, 1);
  
  // caculate first layer
  auto results = lstm_single_layer_direc_npu(input, {h, c}, params, hasBiases, 
      numLayers, dropout, train, bidirectional, batchFirst, false); 

  // get w/ b/ h/ c of twice layer
  at::Tensor weight2Layer;
  at::Tensor bias2Layer;
  at::Tensor h2layer = hx[0].slice(0, 1, 2);
  at::Tensor c2layer = hx[1].slice(0, 1, 2);
  std::tie(weight2Layer, bias2Layer) = get_wb_double_layer_or_bidirec(input, params, hasBiases);
  
  // output of first layer as input of second layer
  at::Tensor input2Layer = std::get<0>(results);
  
  at::Tensor seqMask = at::empty({0}, input.options());
  // caculate output of second layer
  auto results2Layer = NPUNativeFunctions::npu_lstm(input2Layer, weight2Layer, bias2Layer, seqMask, h2layer, c2layer, 
      hasBiases, numLayers, dropout, train, bidirectional, batchFirst, false, false);
  at::Tensor thOutput2Layer = at::unsqueeze(results2Layer[1][numStep-1], 0);
  at::Tensor tcOutput2Layer = at::unsqueeze(results2Layer[2][numStep-1], 0);
  at::Tensor th = at::cat({std::get<1>(results), thOutput2Layer}, 0);
  at::Tensor tc = at::cat({std::get<2>(results), tcOutput2Layer}, 0); 

  return std::tie(results2Layer[0], th, tc); 
}

tuple<at::Tensor, at::Tensor, at::Tensor> lstm_double_layer_bidirec_npu(
    const at::Tensor& input,
    at::TensorList hx,
    at::TensorList params,
    bool hasBiases,
    int64_t numLayers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batchFirst) {
  int64_t numStep = input.size(0);
    
  // get h and c of first layer 
  at::Tensor hL0 = hx[0].slice(0, 0, 2);
  at::Tensor cL0 = hx[1].slice(0, 0, 2); 
  
  // get h and c of second layer
  at::Tensor hL1 = hx[0].slice(0, 2, 4);
  at::Tensor cL1 = hx[1].slice(0, 2, 4);  

  // first Single-layer bidirectional LSTM
  auto resultsLayer1 = lstm_single_layer_bidirec_npu(input, {hL0, cL0}, params, hasBiases, 
      numLayers, dropout, train, bidirectional, batchFirst);     
  
  // second Single-layer bidirectional LSTM, output of Single-layer bidirectional LSTM as input of second layer  
  at::Tensor inputLayer2 = std::get<0>(resultsLayer1);
  at::Tensor y;
  at::Tensor h;
  at::Tensor c;    
  if(hasBiases){
    std::tie(y, h, c) = lstm_single_layer_bidirec_npu(inputLayer2, {hL1, cL1}, params.slice(8, 8), 
        hasBiases, numLayers, dropout, train, bidirectional, batchFirst);
  } else {
    std::tie(y, h, c) = lstm_single_layer_bidirec_npu(inputLayer2, {hL1, cL1}, params.slice(4, 4), 
        hasBiases, numLayers, dropout, train, bidirectional, batchFirst);
  }                   

  at::Tensor th = at::cat({std::get<1>(resultsLayer1), h}, 0);
  at::Tensor tc = at::cat({std::get<2>(resultsLayer1), c}, 0);  
  return std::tie(y, th, tc);                         
}

tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::lstm(
    const at::Tensor& _input,
    at::TensorList hx,
    at::TensorList params,
    bool hasBiases,
    int64_t numLayers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batchFirst) {
  // The operator of DynamicRnn only supports the T axis as the first axis.
  auto input = batchFirst ? _input.transpose(0, 1) : _input;
  at::Tensor y;
  at::Tensor h;
  at::Tensor c;
  
  // single layer
  if(numLayers == 1){
    if(!bidirectional){
      std::tie(y, h, c) = lstm_single_layer_direc_npu(input, hx, params, hasBiases, numLayers, 
          dropout, train, bidirectional, batchFirst, false);
    } else {
      std::tie(y, h, c) = lstm_single_layer_bidirec_npu(input, hx, params, hasBiases, numLayers, 
          dropout, train, bidirectional, batchFirst);
    }
  }

  // double layer
  if(numLayers == 2){
    if(!bidirectional) {
      std::tie(y, h, c) = lstm_double_layer_direc_npu(input, hx, params, hasBiases, numLayers, 
          dropout, train, bidirectional, batchFirst);
    } else {
      std::tie(y, h, c) = lstm_double_layer_bidirec_npu(input, hx, params, hasBiases, numLayers, 
          dropout, train, bidirectional, batchFirst);
    }
  }
    
  // the Bacth axis of output should be first axis when batchFirst is True!
  auto output = batchFirst ? y.transpose(0, 1) : y;
  return std::tie(output, h, c);
}

at::Tensor get_mask(const at::Tensor& input, const at::Tensor& batchSizes, const at::Tensor& h, int64_t maxLen){
  // caculate lengths, but input expected to be sorted
  std::vector<int64_t> lens;
  for (int64_t i = 0; i < input.size(1); ++i){
    auto batchSizesTemp = at::sub(batchSizes , i);
    auto batchSizesBool = at::gt(batchSizesTemp, 0); 
    auto batchSizesInt = batchSizesBool.to(at::ScalarType::Int);
    auto coutLen = at::sum(batchSizesInt, at::ScalarType::Int);
    int64_t len = coutLen.item().toInt();
    lens.emplace_back(len);
  }
  at::Tensor length = CalcuOpUtil::copy_tensor_host_to_device(
      at::from_blob(lens.data(), {lens.size()}, at::kLong));    
  
  c10::SmallVector<at::Tensor, N> maskList;
  // Slice by T axis
  for (int64_t i = 0; i < maxLen; ++i) {    
    // cacl mask
    at::Tensor maskTemp1 = at::gt(length, i);
    at::Tensor maskTemp2 = maskTemp1.reshape({1, input.size(1), 1});
     
    // mask need to be expanded to (1,batch_size,hidden_size)
    at::Tensor maskExpand = maskTemp2.expand({1, input.size(1), h.size(2)});
    maskList.emplace_back(maskExpand);
  }
  
  // mask mast be half
  at::Tensor mask = at::cat(maskList, 0).to(at::ScalarType::Half);

  return mask;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_onelayer_direc_packseq(
    const at::Tensor& data, const at::Tensor& batchSizes, at::TensorList hx,
    at::TensorList params, bool hasBiases,
    int64_t numLayers, double dropoutP, bool train, bool bidirectional) {
  // length of T axis
  int64_t t_size = batchSizes.numel();
  if (t_size == 0) {
    AT_ERROR("lstm_onelayer_direc_packseq: t_size is zero!");
  }

  // T * B **
  at::Tensor input = data.reshape({t_size, data.size(0)/t_size, data.size(1)});

  // batch_first is false
  bool batchFirst = false;

  // get init_h, init_c 
  at::Tensor h = hx[0];
  at::Tensor c = hx[1];
  
  int64_t numStep = input.size(0);
  
  // get weight
  at::Tensor ihWeight = params[0];
  at::Tensor hhWeight = params[1];	
  at::Tensor weight = at::cat({ihWeight, hhWeight}, 1).t().to(input.dtype());
  
  // get bias
  at::Tensor bias = at::zeros(weight.size(1), weight.options());
  if (hasBiases) {
    bias = at::add(params[2], params[3]).to(input.dtype());
  }

  int64_t maxLen = input.size(0);

  at::Tensor mask = get_mask(input, batchSizes, h, maxLen);
  auto results = NPUNativeFunctions::npu_lstm(input, weight, bias, mask, h, c, hasBiases, numLayers, 
      dropoutP, train, bidirectional, false, true, false);  
    
  at::Tensor thOutput = at::unsqueeze(results[1][numStep-1], 0);
  at::Tensor tcOutput = at::unsqueeze(results[2][numStep-1], 0);
  
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(results[0], thOutput, tcOutput);  
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_onelayer_bidirec_packseq(
    const at::Tensor& data, const at::Tensor& batchSizes, at::TensorList hx,
    at::TensorList params, bool hasBiases,
    int64_t numLayers, double dropoutP, bool train, bool bidirectional) {
  // length of T axis
  int64_t t_size = batchSizes.numel();
  if (t_size == 0) {
    AT_ERROR("lstm_onelayer_bidirec_packseq: t_size is zero!");
  }

  // T * B **
  at::Tensor input = data.reshape({t_size, data.size(0)/t_size, data.size(1)});

  // batch_first is false
  bool batchFirst = false;

  // get h and c of forward direction
  at::Tensor h = hx[0].slice(0, 0, 1);
  at::Tensor c = hx[1].slice(0, 0, 1);

  auto resultsForward = lstm_onelayer_direc_packseq(data, batchSizes, {h, c}, params, hasBiases,
      numLayers, dropoutP, train, bidirectional);

  // get w/ b/ h/ c of backward direction
  at::Tensor hBack = hx[0].slice(0, 1, 2);
  at::Tensor cBack = hx[1].slice(0, 1, 2);
  
  at::Tensor weightBack;
  at::Tensor biasBack;
  std::tie(weightBack, biasBack) = get_wb_double_layer_or_bidirec(input, params, hasBiases);

  int64_t maxLen = input.size(0);

  at::Tensor mask = get_mask(input, batchSizes, h, maxLen);
  // caculate forward direction, direction of attr is REDIRECTIONAL
  auto resultsBackward = NPUNativeFunctions::npu_lstm(input, weightBack, biasBack, mask, hBack, cBack, 
      hasBiases, numLayers, dropoutP, train, bidirectional, batchFirst, true, true); 

  // get the first dimension of the T-axis when caculate reverse direction	
  at::Tensor thOutput = at::unsqueeze(resultsBackward[1][0], 0);
  at::Tensor tcOutput = at::unsqueeze(resultsBackward[2][0], 0);
  
  at::Tensor y = at::cat({std::get<0>(resultsForward), resultsBackward[0]}, 2); 
  at::Tensor hOut = at::cat({std::get<1>(resultsForward), thOutput}, 0);
  at::Tensor cOut = at::cat({std::get<2>(resultsForward), tcOutput}, 0);

  return std::tie(y, hOut, cOut);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_double_layer_direc_packseq(
    const at::Tensor& data, const at::Tensor& batchSizes, at::TensorList hx,
    at::TensorList params, bool hasBiases,
    int64_t numLayers, double dropoutP, bool train, bool bidirectional) {
  // length of T axis
  int64_t t_size = batchSizes.numel();
  if (t_size == 0) {
    AT_ERROR("lstm_double_layer_direc_packseq: t_size is zero!");
  }

  // T * B **
  at::Tensor input = data.reshape({t_size, data.size(0)/t_size, data.size(1)});

  // batch_first is false
  bool batchFirst = false;

  // get h and c of forward direction
  at::Tensor h = hx[0].slice(0, 0, 1);
  at::Tensor c = hx[1].slice(0, 0, 1);

  int64_t numStep = input.size(0);

  auto results = lstm_onelayer_direc_packseq(data, batchSizes, {h, c}, params, hasBiases,
      numLayers, dropoutP, train, bidirectional);

  // get w/ b/ h/ c of twice layer
  at::Tensor weight2Layer;
  at::Tensor bias2Layer;
  at::Tensor h2layer = hx[0].slice(0, 1, 2);
  at::Tensor c2layer = hx[1].slice(0, 1, 2);
  std::tie(weight2Layer, bias2Layer) = get_wb_double_layer_or_bidirec(input, params, hasBiases);

  int64_t maxLen = input.size(0);
  at::Tensor mask = get_mask(input, batchSizes, h, maxLen);

  // output of first layer as input of second layer
  at::Tensor input2Layer = std::get<0>(results);

  // caculate output of second layer
  auto results2Layer = NPUNativeFunctions::npu_lstm(input2Layer, weight2Layer, bias2Layer, mask, h2layer, c2layer, 
      hasBiases, numLayers, dropoutP, train, bidirectional, batchFirst, true, false);
  at::Tensor thOutput2Layer = at::unsqueeze(results2Layer[1][numStep-1], 0);
  at::Tensor tcOutput2Layer = at::unsqueeze(results2Layer[2][numStep-1], 0);
  at::Tensor th = at::cat({std::get<1>(results), thOutput2Layer}, 0);
  at::Tensor tc = at::cat({std::get<2>(results), tcOutput2Layer}, 0); 

  return std::tie(results2Layer[0], th, tc);  
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_double_layer_bidirec_packseq(
    const at::Tensor& data, const at::Tensor& batchSizes, at::TensorList hx,
    at::TensorList params, bool hasBiases,
    int64_t numLayers, double dropoutP, bool train, bool bidirectional) {
  // length of T axis
  int64_t t_size = batchSizes.numel();
  TORCH_CHECK(t_size > 0, "batchSizes can not be empty.");
  
  // T * B **
  at::Tensor input = data.reshape({t_size, data.size(0)/t_size, data.size(1)});

  // batch_first is false
  bool batchFirst = false;
  
  // get h and c of first layer 
  at::Tensor hL0 = hx[0].slice(0, 0, 2);
  at::Tensor cL0 = hx[1].slice(0, 0, 2); 
  
  // get h and c of second layer
  at::Tensor hL1 = hx[0].slice(0, 2, 4);
  at::Tensor cL1 = hx[1].slice(0, 2, 4);  

  // first Single-layer bidirectional LSTM
  auto resultsLayer1 = lstm_onelayer_bidirec_packseq(data, batchSizes, {hL0, cL0}, params, hasBiases, 
      numLayers, dropoutP, train, bidirectional);     

  // second Single-layer bidirectional LSTM, output of Single-layer bidirectional LSTM as input of second layer  
  at::Tensor inputLayer2 = std::get<0>(resultsLayer1);
  at::Tensor dataLayer2 = inputLayer2.contiguous().view({-1, inputLayer2.size(2)});
  at::Tensor y;
  at::Tensor h;
  at::Tensor c;
  if(hasBiases){
    std::tie(y, h, c) = lstm_onelayer_bidirec_packseq(dataLayer2, batchSizes, {hL1, cL1}, params.slice(8, 8), 
        hasBiases, numLayers, dropoutP, train, bidirectional);
  } else {
    std::tie(y, h, c) = lstm_onelayer_bidirec_packseq(dataLayer2, batchSizes, {hL1, cL1}, params.slice(4, 4), 
        hasBiases, numLayers, dropoutP, train, bidirectional);
  }
  
  at::Tensor th = at::cat({std::get<1>(resultsLayer1), h}, 0);
  at::Tensor tc = at::cat({std::get<2>(resultsLayer1), c}, 0);  
  return std::tie(y, th, tc);         
 
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::lstm(
    const at::Tensor& data, const at::Tensor& batchSizes_, at::TensorList hx,
    at::TensorList params, bool hasBiases,
    int64_t numLayers, double dropoutP, bool train, bool bidirectional) {
  at::Tensor batchSizes = batchSizes_.to("cpu");
  at::Tensor y;
  at::Tensor h;
  at::Tensor c;

  // single layer
  if(numLayers == 1){
    if(!bidirectional){
      std::tie(y, h, c) = lstm_onelayer_direc_packseq(data, batchSizes, hx, params, hasBiases, 
          numLayers, dropoutP, train, bidirectional);
    } else {
      std::tie(y, h, c) = lstm_onelayer_bidirec_packseq(data, batchSizes, hx, params, hasBiases, 
          numLayers, dropoutP, train, bidirectional);
    }
  }

  // double layer
  if(numLayers == 2) {
    if(!bidirectional){
      std::tie(y, h, c) = lstm_double_layer_direc_packseq(data, batchSizes, hx, params, hasBiases, 
          numLayers, dropoutP, train, bidirectional);
    } else {
      std::tie(y, h, c) = lstm_double_layer_bidirec_packseq(data, batchSizes, hx, params, hasBiases, 
          numLayers, dropoutP, train, bidirectional);
    }
  } 
  return std::tie(y, h, c);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> lstm_backward_out_npu(
    at::Tensor& dw, 
    at::Tensor& db, 
    at::Tensor& dx, 
    at::Tensor& dht, 
    at::Tensor& dct,
    const at::Tensor& x, 
    const at::Tensor& w, 
    const at::Tensor& b, 
    const at::Tensor& init_h, 
    const at::Tensor& init_c, 
    const at::Tensor& dy, 
    const at::Tensor& dh, 
    const at::Tensor& dc,
    const at::Tensor& y, 
    const at::Tensor& h, 
    const at::Tensor& c, 
    const at::Tensor& i,
    const at::Tensor& j, 
    const at::Tensor& f, 
    const at::Tensor& o, 
    const at::Tensor& tanhc) {
  
  at::Tensor seq_length = at::zeros({}, x.options());
  at::Tensor mask = at::zeros({}, x.options().dtype(at::kByte));
  at::Tensor wci = at::zeros({}, x.options());
  at::Tensor wcf = at::zeros({}, x.options());
  at::Tensor wco = at::zeros({}, x.options());

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

  return std::tuple< at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> {dx, dw, db, dht, dct};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> NPUNativeFunctions::npu_lstm_backward(
    const c10::optional<at::Tensor>& grady_opt,
    const c10::optional<at::Tensor>& gradh_opt,
    const c10::optional<at::Tensor>& gradc_opt,
    const at::Tensor& input, 
    const at::Tensor& weight,
    const at::Tensor& bias, 
    const at::Tensor& init_h,
    const at::Tensor& init_c,
    const at::Tensor& y, 
    const at::Tensor& h, 
    const at::Tensor& c, 
    const at::Tensor& i, 
    const at::Tensor& j, 
    const at::Tensor& f, 
    const at::Tensor& o, 
    const at::Tensor& tanhc) { 
  const at::Tensor& grady = c10::value_or_else(grady_opt, [] {return at::Tensor();});
  const at::Tensor& gradh = c10::value_or_else(gradh_opt, [] {return at::Tensor();});
  const at::Tensor& gradc = c10::value_or_else(gradc_opt, [] {return at::Tensor();});

  at::Tensor inh = at::squeeze(init_h, 0);
  at::Tensor inc = at::squeeze(init_c, 0);

  at::Tensor grad_input = OpPreparation::ApplyTensor(input); 
  at::Tensor grad_weight = OpPreparation::ApplyTensor(weight);
  at::Tensor grad_bias = OpPreparation::ApplyTensor(bias);
  at::Tensor grad_ht = OpPreparation::ApplyTensor(init_h);
  at::Tensor grad_ct = OpPreparation::ApplyTensor(init_c);
  
  auto grad_y = grady.defined() ? grady : at::zeros(y.sizes(), y.options());
  auto grad_h = gradh.defined() ? gradh[input.size(0)-1] : at::zeros(inh.sizes(), h.options());
  auto grad_c = gradc.defined() ? gradc[input.size(0)-1] : at::zeros(inc.sizes(), c.options());

  lstm_backward_out_npu(grad_weight, grad_bias, grad_input, grad_ht, grad_ct, input, weight,
                        bias, inh, inc, grad_y, grad_h, grad_c, y, h, c, i, j, f, o, tanhc);

  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> {grad_input, grad_weight, grad_bias, grad_ht, grad_ct};
}

class NPULstmFunction : public torch::autograd::Function<NPULstmFunction> {
public:
  static tensor_list forward(AutogradContext *ctx,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& seqMask,
    const at::Tensor& h,
    const at::Tensor& c,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first,
    bool flagSeq,
    bool flagDirection) {
    at::AutoNonVariableTypeMode g;
    auto result = npu_lstm_npu(
        input,
        weight,
        bias,
        seqMask,
        h,
        c,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        flagSeq,
        flagDirection);
    auto result0 = result[0];
    auto result1 = result[1];
    auto result2 = result[2];
    auto result3 = result[3];
    auto result4 = result[4];
    auto result5 = result[5];
    auto result6 = result[6];
    auto result7 = result[7];
    ctx->save_for_backward({input,
        weight,
        bias,
        h,
        c});
    ctx->saved_data["res0"] = result0;
    ctx->saved_data["res1"] = result1;
    ctx->saved_data["res2"] = result2;
    ctx->saved_data["res3"] = result3;
    ctx->saved_data["res4"] = result4;
    ctx->saved_data["res5"] = result5;
    ctx->saved_data["res6"] = result6;
    ctx->saved_data["res7"] = result7;
    return result;
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];
    auto h = saved[3];
    auto c = saved[4];
    auto result0 = ctx->saved_data["res0"].toTensor();
    auto result1 = ctx->saved_data["res1"].toTensor();
    auto result2 = ctx->saved_data["res2"].toTensor();
    auto result3 = ctx->saved_data["res3"].toTensor();
    auto result4 = ctx->saved_data["res4"].toTensor();
    auto result5 = ctx->saved_data["res5"].toTensor();
    auto result6 = ctx->saved_data["res6"].toTensor();
    auto result7 = ctx->saved_data["res7"].toTensor();

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> result = NPUNativeFunctions::npu_lstm_backward(
        grad_outputs[0],
        grad_outputs[1],
        grad_outputs[2],
        input,
        weight,
        bias,
        h,
        c,
        result0,
        result1,
        result2,
        result3,
        result4,
        result5,
        result6,
        result7);
    tensor_list output = {std::get<0>(result),
                          std::get<1>(result),
                          std::get<2>(result),
                          at::Tensor(),
                          std::get<3>(result),
                          std::get<4>(result),
                          at::Tensor(),
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

std::vector<at::Tensor> NPUNativeFunctions::npu_lstm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& seqMask,
    const at::Tensor& h,
    const at::Tensor& c,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batch_first,
    bool flagSeq,
    bool flagDirection){
  return NPULstmFunction::apply(
      input,
      weight,
      bias,
      seqMask,
      h,
      c,
      has_biases,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_first,
      flagSeq,
      flagDirection);
}

} // namespace native
} // namespace at_npu
