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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::_ctc_loss(
    const at::Tensor& logProbs,
    const at::Tensor& targets,
    at::IntArrayRef inputLengths,
    at::IntArrayRef targetLengths,
    int64_t blank,
    bool zeroInfinity) {
  at::Tensor logProbsNeed = logProbs;
  if (logProbs.scalar_type() == at::ScalarType::Half) {
    logProbsNeed = NPUNativeFunctions::npu_dtype_cast(logProbsNeed, at::ScalarType::Float);
  }
  
  // Aicore supports only the int type
  at::Tensor targetsCast = targets;
  if(targets.scalar_type() == at::ScalarType::Long){
    targetsCast = NPUNativeFunctions::npu_dtype_cast(targetsCast, at::ScalarType::Int);
  }
  
  // IntArrayRef to Tensor
  auto inputLengthsTensor = at::tensor(inputLengths, targetsCast.options());
  auto targetLengthsTensor = at::tensor(targetLengths, targetsCast.options());
  
  int64_t maxLength = 0;
  if (targetsCast.dim() == 2) {
    maxLength = targetsCast.size(1);  
  } else if (targetsCast.dim() == 1) {
    for (auto &i : targetLengths) {
      if (i > maxLength) {
        maxLength = i;
      }
    }
  }
  
  auto shape = logProbs.sizes();
  
  auto blankNew = blank + maxLength * shape[2];
  
  // calculate the output size
  auto outputSizes = ctc_loss_npu_output_size(logProbs, targetsCast, targetLengths, maxLength);

  // construct the output tensor of the NPU
  at::Tensor negLogLikelihood = OpPreparation::ApplyTensorWithFormat(
      std::get<0>(outputSizes),
      logProbsNeed.options(),
      CalcuOpUtil::GetTensorNpuFormat(logProbsNeed));
  
  at::Tensor logAlpha = OpPreparation::ApplyTensorWithFormat(
      std::get<1>(outputSizes),
      logProbsNeed.options(),
      CalcuOpUtil::GetTensorNpuFormat(logProbsNeed));

  // calculate the output result of the NPU 
  OpCommand cmd;
  if (targetsCast.dim() == 2) {
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
  } else if (targetsCast.dim() == 1) {
    cmd.Name("CTCLossV2")
      .Input(logProbsNeed)
      .Input(targetsCast)
      .Input(inputLengthsTensor)
      .Input(targetLengthsTensor)
      .Output(negLogLikelihood)
      .Output(logAlpha)
      .Attr("blank", blankNew)
      .Attr("zero_infinity", zeroInfinity)
      .Run();  
  }

  
  if (logProbs.scalar_type() == at::ScalarType::Half) {
    negLogLikelihood = NPUNativeFunctions::npu_dtype_cast(negLogLikelihood, at::ScalarType::Half);
    logAlpha = NPUNativeFunctions::npu_dtype_cast(logAlpha, at::ScalarType::Half);
  }  

  return std::tuple<at::Tensor, at::Tensor>(negLogLikelihood, logAlpha);
}

at::Tensor NPUNativeFunctions::ctc_loss(
    const at::Tensor& logProbs,
    const at::Tensor& targets,
    at::IntArrayRef inputLengths,
    at::IntArrayRef targetLengths,
    int64_t blank,
    int64_t reduction,
    bool zeroInfinity) {
  at::Tensor res = std::get<0>(at::_ctc_loss(
      logProbs, 
      targets, 
      inputLengths, 
      targetLengths, 
      blank, 
      zeroInfinity));
  
  if (zeroInfinity) {
    res = at::where(
        res == at::Scalar(std::numeric_limits<double>::infinity()), 
        at::zeros({}, res.options()), 
        res);   
  }

  if (reduction == at::Reduction::Mean) {
    std::vector<int64_t> targetLengthsVector = targetLengths.vec();

    auto targetLengthsTensor = CalcuOpUtil::CopyTensorHostToDevice(
        at::from_blob(targetLengthsVector.data(), {targetLengthsVector.size()}, at::kLong)).clamp_min(1);

    at::Tensor targetLengthsTensor_ = targetLengthsTensor.to(res.dtype()); 

    return (res / targetLengthsTensor_).mean(); 

  } else if (reduction == at::Reduction::Sum) {
    return res.sum();
  }

  return res;
}

at::Tensor NPUNativeFunctions::ctc_loss(
    const at::Tensor& logProbs,
    const at::Tensor& targets,
    const at::Tensor& inputLengths,
    const at::Tensor& targetLengths,
    int64_t blank,
    int64_t reduction,
    bool zeroInfinity) { 
  TORCH_CHECK(isIntegralType(inputLengths.scalar_type(), false), "input_lengths must be integral");
  TORCH_CHECK(isIntegralType(targetLengths.scalar_type(), false), "target_lengths must be integral");

  at::Tensor inputLengthsTensor = inputLengths.to(at::Device(at::kCPU), at::kLong).contiguous();
  at::Tensor targetLengthsTensor = targetLengths.to(at::Device(at::kCPU), at::kLong).contiguous();
  
  at::IntArrayRef inputLengthsList(inputLengthsTensor.data_ptr<int64_t>(), inputLengthsTensor.numel());
  at::IntArrayRef targetLengthsList(targetLengthsTensor.data_ptr<int64_t>(), targetLengthsTensor.numel());
  
  return at::ctc_loss(logProbs, targets, inputLengthsList, targetLengthsList, blank, reduction, zeroInfinity);
}
} // namespace native
} // namespace at_npu