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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::_ctc_loss_backward(
    const at::Tensor& gradOut,
    const at::Tensor& logProbs,
    const at::Tensor& targets,
    at::IntArrayRef inputLengths,
    at::IntArrayRef targetLengths,
    const at::Tensor& negLogLikelihood,
    const at::Tensor& logAlpha,
    int64_t blank,
    bool zeroInfinity) {
  at::Tensor gradOutNeed = gradOut;
  if (gradOut.scalar_type() == at::ScalarType::Half) {
    gradOutNeed = NPUNativeFunctions::npu_dtype_cast(gradOutNeed, at::ScalarType::Float);
  }

  at::Tensor logProbsNeed = logProbs;
  if (logProbs.scalar_type() == at::ScalarType::Half) {
    logProbsNeed = NPUNativeFunctions::npu_dtype_cast(logProbsNeed, at::ScalarType::Float);
  }

  at::Tensor negLogLikelihoodNeed = negLogLikelihood;
  if (negLogLikelihood.scalar_type() == at::ScalarType::Half) {
    negLogLikelihoodNeed = NPUNativeFunctions::npu_dtype_cast(negLogLikelihoodNeed, at::ScalarType::Float);
  }

  at::Tensor logAlphaNeed = logAlpha;
  if (logAlpha.scalar_type() == at::ScalarType::Half) {
    logAlphaNeed = NPUNativeFunctions::npu_dtype_cast(logAlphaNeed, at::ScalarType::Float);
    
  }
  
  at::Tensor targetsCast = targets;
  if(targets.scalar_type() == at::ScalarType::Long){
    targetsCast = NPUNativeFunctions::npu_dtype_cast(targetsCast, at::ScalarType::Int);
  }
  
  auto inputLengthsTensor = at::tensor(inputLengths, targetsCast.options().dtype(at::kInt));
  auto targetLengthsTensor = at::tensor(targetLengths, targetsCast.options().dtype(at::kInt));

  auto outputSize = input_same_output_size(logProbs);

  // construct the output tensor of the NPU
  at::Tensor grad = OpPreparation::ApplyTensor(logProbsNeed, outputSize);
  // calculate the output result of the NPU
  OpCommand cmd;
  cmd.Name("CTCLossV2Grad")
      .Input(gradOutNeed)
      .Input(logProbsNeed)
      .Input(targetsCast)
      .Input(inputLengthsTensor)
      .Input(targetLengthsTensor)      
      .Input(negLogLikelihoodNeed)
      .Input(logAlphaNeed)      
      .Output(grad)
      .Attr("blank", blank)
      .Attr("zero_infinity", zeroInfinity)
      .Run();

  if (gradOut.scalar_type() == at::ScalarType::Half) {
    grad = NPUNativeFunctions::npu_dtype_cast(grad, at::ScalarType::Half);
  }
  
  return grad;
}
} // namespace native
} // namespace at_npu
