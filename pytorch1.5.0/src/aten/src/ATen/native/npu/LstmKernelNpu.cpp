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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> lstm_npu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& seqMask,
    const Tensor& h,
    const Tensor& c,
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

  return std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(
      yOutput, hOutput, cOutput, iOutput, jOutput, fOutput, oOutput, tanhc);
}

tuple<Tensor, Tensor> get_wb_single_layer_direc(
    const Tensor& input,
    TensorList params,
    bool hasBiases) {
  // get weight
  Tensor ihWeight = params[0];
  Tensor hhWeight = params[1];
	
  Tensor weight = at::cat({ihWeight, hhWeight}, 1).t().to(input.dtype());
  
  // get bias
  Tensor bias = at::zeros(weight.size(1), weight.options());
  if (hasBiases) {
    bias = at::add(params[2], params[3]).to(input.dtype());
  }
  return std::tie(weight, bias);
}

tuple<Tensor, Tensor> get_wb_double_layer_or_bidirec(
    const Tensor& input,
    TensorList params,
    bool hasBiases) {
  Tensor weight;
  Tensor bias; 
  if (hasBiases) {
    weight = at::cat({params[4], params[5]}, 1).t().to(input.dtype());
    bias = at::add(params[6], params[7]).to(input.dtype());
  } else {
    weight = at::cat({params[2], params[3]}, 1).t().to(input.dtype());
    bias = at::zeros(weight.size(1), weight.options());
  }
  return std::tie(weight, bias);
}

tuple<Tensor, Tensor, Tensor> lstm_single_layer_direc_npu(
    const Tensor& input,
    TensorList hx,
    TensorList params,
    bool hasBiases,
    int64_t numLayers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batchFirst,
    bool direction) {
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
  
  Tensor seqMask = at::empty({0}, input.options());
  auto results = at::npu_lstm(input, weight, bias, seqMask, h, c, hasBiases, numLayers, dropout, 
      train, bidirectional, batchFirst, false, direction);

  // get the last dimension of the T-axis	
  Tensor thOutput = at::unsqueeze(std::get<1>(results)[numStep-1], 0);
  Tensor tcOutput = at::unsqueeze(std::get<2>(results)[numStep-1], 0);

  return std::tie(std::get<0>(results), thOutput, tcOutput);
}

tuple<Tensor, Tensor, Tensor> lstm_single_layer_bidirec_npu(
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
  // get h and c of forward direction
  Tensor h = hx[0].slice(0, 0, 1);
  Tensor c = hx[1].slice(0, 0, 1);
  
  // caculate forward direction, direction of attr is UNIDIRECTIONAL(npu_lstm need add the attr of direction)
  auto resultsForward = lstm_single_layer_direc_npu(input, {h, c}, params, hasBiases, 
      numLayers, dropout, train, bidirectional, batchFirst, false); 

  // get w/ b/ h/ c of backward direction
  Tensor weightBack;
  Tensor biasBack;
  Tensor hBack = hx[0].slice(0, 1, 2);
  Tensor cBack = hx[1].slice(0, 1, 2);
  std::tie(weightBack, biasBack) = get_wb_double_layer_or_bidirec(input, params, hasBiases);

  Tensor seqMask = at::empty({0}, input.options());
  // caculate forward direction, direction of attr is REDIRECTIONAL
  auto resultsBackward = at::npu_lstm(input, weightBack, biasBack, seqMask, hBack, cBack, 
      hasBiases, numLayers, dropout, train, bidirectional, batchFirst, false, true);
  
  // get the first dimension of the T-axis when caculate reverse direction	
  Tensor thOutput = at::unsqueeze(std::get<1>(resultsBackward)[0], 0);
  Tensor tcOutput = at::unsqueeze(std::get<2>(resultsBackward)[0], 0);

  Tensor y = at::cat({std::get<0>(resultsForward), std::get<0>(resultsBackward)}, 2); 
  Tensor hOut = at::cat({std::get<1>(resultsForward), thOutput}, 0);
  Tensor cOut = at::cat({std::get<2>(resultsForward), tcOutput}, 0);

  return std::tie(y, hOut, cOut);
}

tuple<Tensor, Tensor, Tensor> lstm_double_layer_direc_npu(
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
  // get h and c of first layer
  Tensor h = hx[0].slice(0, 0, 1);
  Tensor c = hx[1].slice(0, 0, 1);
  
  // caculate first layer
  auto results = lstm_single_layer_direc_npu(input, {h, c}, params, hasBiases, 
      numLayers, dropout, train, bidirectional, batchFirst, false); 

  // get w/ b/ h/ c of twice layer
  Tensor weight2Layer;
  Tensor bias2Layer;
  Tensor h2layer = hx[0].slice(0, 1, 2);
  Tensor c2layer = hx[1].slice(0, 1, 2);
  std::tie(weight2Layer, bias2Layer) = get_wb_double_layer_or_bidirec(input, params, hasBiases);
  
  // output of first layer as input of second layer
  Tensor input2Layer = std::get<0>(results);
  
  Tensor seqMask = at::empty({0}, input.options());
  // caculate output of second layer
  auto results2Layer = at::npu_lstm(input2Layer, weight2Layer, bias2Layer, seqMask, h2layer, c2layer, 
      hasBiases, numLayers, dropout, train, bidirectional, batchFirst, false, false);
  Tensor thOutput2Layer = at::unsqueeze(std::get<1>(results2Layer)[numStep-1], 0);
  Tensor tcOutput2Layer = at::unsqueeze(std::get<2>(results2Layer)[numStep-1], 0);
  Tensor th = at::cat({std::get<1>(results), thOutput2Layer}, 0);
  Tensor tc = at::cat({std::get<2>(results), tcOutput2Layer}, 0); 

  return std::tie(std::get<0>(results2Layer), th, tc); 
}

tuple<Tensor, Tensor, Tensor> lstm_npu(
    const Tensor& _input,
    TensorList hx,
    TensorList params,
    bool hasBiases,
    int64_t numLayers,
    double dropout,
    bool train,
    bool bidirectional,
    bool batchFirst) {
  // The operator of DynamicRnn only supports the T axis as the first axis.
  auto input = batchFirst ? _input.transpose(0, 1) : _input;

  Tensor y;
  Tensor h;
  Tensor c;
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
  if((numLayers == 2) && (!bidirectional)) {
    std::tie(y, h, c) = lstm_double_layer_direc_npu(input, hx, params, hasBiases, numLayers, 
        dropout, train, bidirectional, batchFirst);
  }

  // the Bacth axis of output should be first axis when batchFirst is True!
  auto output = batchFirst ? y.transpose(0, 1) : y;  
  return std::tie(output, h, c);
}

Tensor get_mask(const Tensor& input, const Tensor& batchSizes, const Tensor& h, int64_t maxLen){
  // caculate lengths, but input expected to be sorted
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
  
  SmallVector<Tensor, N> maskList;
  // Slice by T axis
  for (int64_t i = 0; i < maxLen; ++i) {    
    // cacl mask
    Tensor maskTemp1 = at::gt(length, i);
    Tensor maskTemp2 = maskTemp1.reshape({1, input.size(1), 1});
     
    // mask need to be expanded to (1,batch_size,hidden_size)
    Tensor maskExpand = maskTemp2.expand({1, input.size(1), h.size(2)});
    maskList.emplace_back(maskExpand);
  }
  
  // mask mast be half
  Tensor mask = at::cat(maskList, 0).to(ScalarType::Half);

  return mask;
}

std::tuple<Tensor, Tensor, Tensor> lstm_onelayer_direc_packseq(
    const Tensor& data, const Tensor& batchSizes, TensorList hx,
    TensorList params, bool hasBiases,
    int64_t numLayers, double dropoutP, bool train, bool bidirectional) {
  // length of T axis
  int64_t t_size = batchSizes.numel();
  if (t_size == 0) {
    AT_ERROR("lstm_onelayer_direc_packseq: t_size is zero!");
  }

  // T * B **
  Tensor input = data.reshape({t_size, data.size(0)/t_size, data.size(1)});

  // batch_first is false
  bool batchFirst = false;

  // get init_h, init_c 
  Tensor h = hx[0];
  Tensor c = hx[1];
  
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

  int64_t maxLen = input.size(0);

  Tensor mask = get_mask(input, batchSizes, h, maxLen);
  auto results = at::npu_lstm(input, weight, bias, mask, h, c, hasBiases, numLayers, 
      dropoutP, train, bidirectional, false, true, false);  
    
  Tensor thOutput = at::unsqueeze(std::get<1>(results)[numStep-1], 0);
  Tensor tcOutput = at::unsqueeze(std::get<2>(results)[numStep-1], 0);
  
  return std::tuple<Tensor, Tensor, Tensor>(std::get<0>(results), thOutput, tcOutput);  
}

std::tuple<Tensor, Tensor, Tensor> lstm_onelayer_bidirec_packseq(
    const Tensor& data, const Tensor& batchSizes, TensorList hx,
    TensorList params, bool hasBiases,
    int64_t numLayers, double dropoutP, bool train, bool bidirectional) {
  // length of T axis
  int64_t t_size = batchSizes.numel();
  if (t_size == 0) {
    AT_ERROR("lstm_onelayer_bidirec_packseq: t_size is zero!");
  }

  // T * B **
  Tensor input = data.reshape({t_size, data.size(0)/t_size, data.size(1)});

  // batch_first is false
  bool batchFirst = false;

  // get h and c of forward direction
  Tensor h = hx[0].slice(0, 0, 1);
  Tensor c = hx[1].slice(0, 0, 1);

  auto resultsForward = lstm_onelayer_direc_packseq(data, batchSizes, {h, c}, params, hasBiases,
      numLayers, dropoutP, train, bidirectional);

  // get w/ b/ h/ c of backward direction
  Tensor hBack = hx[0].slice(0, 1, 2);
  Tensor cBack = hx[1].slice(0, 1, 2);
  
  Tensor weightBack;
  Tensor biasBack;
  std::tie(weightBack, biasBack) = get_wb_double_layer_or_bidirec(input, params, hasBiases);

  int64_t maxLen = input.size(0);

  Tensor mask = get_mask(input, batchSizes, h, maxLen);
  // caculate forward direction, direction of attr is REDIRECTIONAL
  auto resultsBackward = at::npu_lstm(input, weightBack, biasBack, mask, hBack, cBack, 
      hasBiases, numLayers, dropoutP, train, bidirectional, batchFirst, true, true); 

  // get the first dimension of the T-axis when caculate reverse direction	
  Tensor thOutput = at::unsqueeze(std::get<1>(resultsBackward)[0], 0);
  Tensor tcOutput = at::unsqueeze(std::get<2>(resultsBackward)[0], 0);
  
  Tensor y = at::cat({std::get<0>(resultsForward), std::get<0>(resultsBackward)}, 2); 
  Tensor hOut = at::cat({std::get<1>(resultsForward), thOutput}, 0);
  Tensor cOut = at::cat({std::get<2>(resultsForward), tcOutput}, 0);

  return std::tie(y, hOut, cOut);
}

std::tuple<Tensor, Tensor, Tensor> lstm_double_layer_direc_packseq(
    const Tensor& data, const Tensor& batchSizes, TensorList hx,
    TensorList params, bool hasBiases,
    int64_t numLayers, double dropoutP, bool train, bool bidirectional) {
  // length of T axis
  int64_t t_size = batchSizes.numel();
  if (t_size == 0) {
    AT_ERROR("lstm_double_layer_direc_packseq: t_size is zero!");
  }

  // T * B **
  Tensor input = data.reshape({t_size, data.size(0)/t_size, data.size(1)});

  // batch_first is false
  bool batchFirst = false;

  // get h and c of forward direction
  Tensor h = hx[0].slice(0, 0, 1);
  Tensor c = hx[1].slice(0, 0, 1);

  int64_t numStep = input.size(0);

  auto results = lstm_onelayer_direc_packseq(data, batchSizes, {h, c}, params, hasBiases,
      numLayers, dropoutP, train, bidirectional);

  // get w/ b/ h/ c of twice layer
  Tensor weight2Layer;
  Tensor bias2Layer;
  Tensor h2layer = hx[0].slice(0, 1, 2);
  Tensor c2layer = hx[1].slice(0, 1, 2);
  std::tie(weight2Layer, bias2Layer) = get_wb_double_layer_or_bidirec(input, params, hasBiases);

  int64_t maxLen = input.size(0);

  Tensor mask = get_mask(input, batchSizes, h, maxLen);

  // output of first layer as input of second layer
  Tensor input2Layer = std::get<0>(results);

  // caculate output of second layer
  auto results2Layer = at::npu_lstm(input2Layer, weight2Layer, bias2Layer, mask, h2layer, c2layer, 
      hasBiases, numLayers, dropoutP, train, bidirectional, batchFirst, true, false);
  Tensor thOutput2Layer = at::unsqueeze(std::get<1>(results2Layer)[numStep-1], 0);
  Tensor tcOutput2Layer = at::unsqueeze(std::get<2>(results2Layer)[numStep-1], 0);
  Tensor th = at::cat({std::get<1>(results), thOutput2Layer}, 0);
  Tensor tc = at::cat({std::get<2>(results), tcOutput2Layer}, 0); 

  return std::tie(std::get<0>(results2Layer), th, tc);  
}

std::tuple<Tensor, Tensor, Tensor> lstm_npu(
    const Tensor& data, const Tensor& batchSizes, TensorList hx,
    TensorList params, bool hasBiases,
    int64_t numLayers, double dropoutP, bool train, bool bidirectional) {
  Tensor y;
  Tensor h;
  Tensor c;

  // batch_first is false
  bool batchFirst = false;

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
  if((numLayers == 2) && (!bidirectional)) {
    std::tie(y, h, c) = lstm_double_layer_direc_packseq(data, batchSizes, hx, params, hasBiases, 
        numLayers, dropoutP, train, bidirectional);
  } 
  return std::tie(y, h, c);
}

} // namespace native
} // namespace at