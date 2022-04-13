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

Tensor& normal_out_npu(
    Tensor& result,
    const Tensor& mean, 
    double std, 
    Generator* generator) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);

  Tensor resultCopy = result;
  Tensor dtypeCastOfMean = mean;
  if (dtypeCastOfMean.scalar_type() == ScalarType::Half) {
    dtypeCastOfMean = dtypeCastOfMean.to(ScalarType::Float);
    resultCopy = resultCopy.to(ScalarType::Float);
  }

  OpCommand cmd;
  cmd.Name("Normal")
    .Input(dtypeCastOfMean)
    .Input(Scalar(std), ScalarType::Float)
    .Output(resultCopy)
    .Run();

  result.copy_(resultCopy);

  return result;
}

Tensor& normal_out_npu(
    Tensor& result,
    double mean, 
    const Tensor& std, 
    Generator* generator) {
  Tensor resultCopy = result;
  Tensor dtypeCastOfStd = std;
  if (dtypeCastOfStd.scalar_type() == ScalarType::Half) {
    dtypeCastOfStd = dtypeCastOfStd.to(ScalarType::Float);
    resultCopy = resultCopy.to(ScalarType::Float);
  }

  OpCommand cmd;
  cmd.Name("Normal")
    .Input(Scalar(mean), ScalarType::Float)
    .Input(dtypeCastOfStd)
    .Output(resultCopy)
    .Run();

  result.copy_(resultCopy);

  return result;
}

Tensor& normal_out_npu(
    Tensor& result,
    const Tensor& mean, 
    const Tensor& std, 
    Generator* generator) {
  Tensor resultCopy = result;  
  Tensor dtypeCastOfMean = mean;
  Tensor dtypeCastOfStd = std;
  if (dtypeCastOfMean.scalar_type() == ScalarType::Half) {
    dtypeCastOfMean = dtypeCastOfMean.to(ScalarType::Float);
    resultCopy = resultCopy.to(ScalarType::Float);
  }
  if (dtypeCastOfStd.scalar_type() == ScalarType::Half) {
    dtypeCastOfStd = dtypeCastOfStd.to(ScalarType::Float);
  }
  OpCommand cmd;
  cmd.Name("Normal")
    .Input(dtypeCastOfMean)
    .Input(dtypeCastOfStd)
    .Output(resultCopy)
    .Run();

  result.copy_(resultCopy);

  return result;
}

Tensor& normal_out_npu(
    Tensor& result,
    double mean, 
    double std, 
    IntArrayRef size,
    Generator* generator) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);

  // the op of PTNormalFloatFloat only support format of ND
  Tensor formatCastOfResult = result.npu_format_cast(ACL_FORMAT_ND);
  if (formatCastOfResult.scalar_type() == ScalarType::Half) {
    formatCastOfResult = formatCastOfResult.to(ScalarType::Float);
  }

  Tensor meanTensor = OpPreparation::ApplyTensor(size, formatCastOfResult.options(), result);
  meanTensor.fill_(mean);
  OpCommand cmd;
  cmd.Name("Normal")
    .Input(meanTensor)
    .Input(Scalar(std), ScalarType::Float)
    .Output(formatCastOfResult)
    .Run();

  result.copy_(formatCastOfResult);

  return result;
}

Tensor normal_npu(
    const Tensor& mean, 
    double std, 
    Generator* generator) {
  Tensor result = OpPreparation::ApplyTensor(mean);
  normal_out_npu(result, mean, std, generator);

  return result;
}

Tensor normal_npu(
    double mean, 
    const Tensor& std, 
    Generator* generator) {
  Tensor result = OpPreparation::ApplyTensor(std);
  normal_out_npu(result, mean, std, generator);

  return result;
}

Tensor normal_npu(
    const Tensor& mean, 
    const Tensor& std, 
    Generator* generator) {
  Tensor result = OpPreparation::ApplyTensor(mean);
  normal_out_npu(result, mean, std, generator);

  return result;
}

Tensor normal_npu(
    double mean, 
    double std, 
    IntArrayRef size,
    Generator* generator,
    const TensorOptions& options) {
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      size, options, ACL_FORMAT_ND);

  // calculate the output result of the NPU
  normal_out_npu(result, mean, std, size, generator);

  return result;
}

Tensor& normal_npu_(
    Tensor& self,
    double mean,
    double std,
    Generator* generator) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = normal_out_npu(contiguousSelf, mean, std, contiguousSelf.sizes(), generator);
    NpuUtils::format_fresh_view(self, result);
  } else {
    normal_out_npu(self, mean, std, self.sizes(), generator);
  }

  return self;
}

} // namespace native
} // namespace at