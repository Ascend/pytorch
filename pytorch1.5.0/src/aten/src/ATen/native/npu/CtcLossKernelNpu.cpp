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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

std::tuple<Tensor, Tensor> ctc_loss_npu(
    const Tensor& logProbs,
    const Tensor& targets,
    IntArrayRef inputLengths,
    IntArrayRef targetLengths,
    int64_t blank,
    bool zeroInfinity) {
  Tensor logProbsNeed = logProbs;
  if (logProbs.scalar_type() == ScalarType::Half) {
    logProbsNeed = logProbsNeed.to(ScalarType::Float);
  }
  
  //Aicore supports only the int type
  Tensor targetsCast = targets;
  if(targets.scalar_type() == ScalarType::Long){
    targetsCast = targetsCast.to(ScalarType::Int);
  }
  
  // IntArrayRef to Tensor
  auto inputLengthsTensor = at::tensor(inputLengths, targetsCast.options());
  auto targetLengthsTensor = at::tensor(targetLengths, targetsCast.options());
  
  // calculate the output size
  auto outputSizes = ctc_loss_npu_output_size(logProbs, targetsCast, targetLengths);

  // construct the output tensor of the NPU
  Tensor negLogLikelihood = at::empty_with_format(
      std::get<0>(outputSizes),
      logProbsNeed.options(),
      CalcuOpUtil::get_tensor_npu_format(logProbsNeed));
  
  Tensor logAlpha = at::empty_with_format(
      std::get<1>(outputSizes),
      logProbsNeed.options(),
      CalcuOpUtil::get_tensor_npu_format(logProbsNeed));

  // calculate the output result of the NPU 
  OpCommand cmd;
  cmd.Name("CTCLossV2")
      .Input(logProbsNeed)
      .Input(targetsCast)
      .Input(inputLengthsTensor)
      .Input(targetLengthsTensor)
      .Output(negLogLikelihood)
      .Output(logAlpha)
      .Attr("blank", blank)
      .Attr("zero_infinity", zeroInfinity)
      .Run();
  
  if (logProbs.scalar_type() == ScalarType::Half) {
    negLogLikelihood = negLogLikelihood.npu_dtype_cast(ScalarType::Half);
    logAlpha = logAlpha.to(ScalarType::Half);
  }  

  return std::tuple<Tensor, Tensor>(negLogLikelihood, logAlpha);
}

Tensor ctc_loss_npu(
    const Tensor& logProbs,
    const Tensor& targets,
    IntArrayRef inputLengths,
    IntArrayRef targetLengths,
    int64_t blank,
    int64_t reduction,
    bool zeroInfinity) {
  Tensor res = std::get<0>(at::_ctc_loss(
      logProbs, 
      targets, 
      inputLengths, 
      targetLengths, 
      blank, 
      zeroInfinity));
  
  if (zeroInfinity) {
    res = at::where(
      res == Scalar(std::numeric_limits<double>::infinity()), 
      at::zeros({}, res.options()), 
      res);   
  }

  if (reduction == at::Reduction::Mean) {
    std::vector<int64_t> targetLengthsVector = targetLengths.vec();

    auto targetLengthsTensor = CalcuOpUtil::copy_tensor_host_to_device(
        from_blob(targetLengthsVector.data(), {targetLengthsVector.size()}, at::kLong)).clamp_min(1);

    Tensor targetLengthsTensor_ = targetLengthsTensor.to(res.dtype()); 
    return (res / targetLengthsTensor_).mean(); 

  } else if (reduction == at::Reduction::Sum) {
    return res.sum();
  }

  return res;
}

Tensor ctc_loss_npu(
    const Tensor& logProbs,
    const Tensor& targets,
    const Tensor& inputLengths,
    const Tensor& targetLengths,
    int64_t blank,
    int64_t reduction,
    bool zeroInfinity) { 
  TORCH_CHECK(isIntegralType(inputLengths.scalar_type(), /*includeBool=*/false), "input_lengths must be integral");
  TORCH_CHECK(isIntegralType(targetLengths.scalar_type(), /*includeBool=*/false), "target_lengths must be integral");

  Tensor inputLengthsTensor = inputLengths.to(Device(at::kCPU), at::kLong).contiguous();
  Tensor targetLengthsTensor = targetLengths.to(Device(at::kCPU), at::kLong).contiguous();
  
  IntArrayRef inputLengthsList(inputLengthsTensor.data_ptr<int64_t>(), inputLengthsTensor.numel());
  IntArrayRef targetLengthsList(targetLengthsTensor.data_ptr<int64_t>(), targetLengthsTensor.numel());
  
  
  return at::ctc_loss(logProbs, targets, inputLengthsList, targetLengthsList, blank, reduction, zeroInfinity);
}

} // namespace native
} // namespace at
