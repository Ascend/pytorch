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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor ctc_loss_backward_npu(
    const Tensor& gradOut,
    const Tensor& logProbs,
    const Tensor& targets,
    IntArrayRef inputLengths,
    IntArrayRef targetLengths,
    const Tensor& negLogLikelihood,
    const Tensor& logAlpha,
    int64_t blank,
    bool zeroInfinity) {
  Tensor gradOutNeed = gradOut;
  if (gradOut.scalar_type() == ScalarType::Half) {
    gradOutNeed = gradOutNeed.to(ScalarType::Float);
  }

  Tensor logProbsNeed = logProbs;
  if (logProbs.scalar_type() == ScalarType::Half) {
    logProbsNeed = logProbsNeed.to(ScalarType::Float);
  }

  Tensor negLogLikelihoodNeed = negLogLikelihood;
  if (negLogLikelihood.scalar_type() == ScalarType::Half) {
    negLogLikelihoodNeed = negLogLikelihoodNeed.to(ScalarType::Float);
  }

  Tensor logAlphaNeed = logAlpha;
  if (logAlpha.scalar_type() == ScalarType::Half) {
    logAlphaNeed = logAlphaNeed.to(ScalarType::Float);
  }
  
  Tensor targetsCast = targets;
  if(targets.scalar_type() == ScalarType::Long){
    targetsCast = targetsCast.to(ScalarType::Int);
  }
  
  auto inputLengthsTensor = at::tensor(inputLengths, targetsCast.options().dtype(at::kInt));
  auto targetLengthsTensor = at::tensor(targetLengths, targetsCast.options().dtype(at::kInt));

  auto outputSize = input_same_output_size(logProbs);

  // construct the output tensor of the NPU
  Tensor grad = OpPreparation::ApplyTensor(logProbsNeed, outputSize);
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

  if (gradOut.scalar_type() == ScalarType::Half) {
    grad = grad.to(ScalarType::Half);
  }
  
  return grad;
}
} // namespace native
} // namespace at
