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

#include <torch/script.h>
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& normal_tensor_float_out_npu(
    const Tensor& mean, 
    double std, 
    c10::optional<Generator> generator,
    Tensor& result) {
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
    .Input(std, ScalarType::Float)
    .Output(resultCopy)
    .Run();

  result.copy_(resultCopy);

  return result;
}

Tensor& normal_float_tensor_out_npu(
    double mean, 
    const Tensor& std, 
    c10::optional<Generator> generator,
    Tensor& result) {
  Tensor resultCopy = result;
  Tensor dtypeCastOfStd = std;
  if (dtypeCastOfStd.scalar_type() == ScalarType::Half) {
    dtypeCastOfStd = dtypeCastOfStd.to(ScalarType::Float);
    resultCopy = resultCopy.to(ScalarType::Float);
  }

  OpCommand cmd;
  cmd.Name("Normal")
    .Input(mean, ScalarType::Float)
    .Input(dtypeCastOfStd)
    .Output(resultCopy)
    .Run();

  result.copy_(resultCopy);

  return result;
}

Tensor& normal_tensor_tensor_out_npu(
    const Tensor& mean, 
    const Tensor& std, 
    c10::optional<Generator> generator,    
    Tensor& result) {
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

Tensor& normal_float_float_out_npu(
    double mean, 
    double std, 
    IntArrayRef size,
    c10::optional<Generator> generator, 
    Tensor& result) {
  TORCH_CHECK(std > 0.0, "normal_ expects std > 0.0, but found std=", std);

  //the op of PTNormalFloatFloat only support format of ND
  Tensor formatCastOfResult = result.npu_format_cast(ACL_FORMAT_ND);
  if (formatCastOfResult.scalar_type() == ScalarType::Half) {
    formatCastOfResult = formatCastOfResult.to(ScalarType::Float);
  }
  
  Tensor meanTensor = OpPreparation::ApplyTensor(size, result.options(), result);
  meanTensor.fill_(mean);
  OpCommand cmd;
  cmd.Name("Normal")
    .Input(meanTensor)
    .Input(std, ScalarType::Float)
    .Output(formatCastOfResult)
    .Run();

  result.copy_(formatCastOfResult);

  return result;
}

Tensor normal_tensor_float_npu(
    const Tensor& mean, 
    double std, 
    c10::optional<Generator> generator) {
  Tensor result = OpPreparation::ApplyTensor(mean);
  normal_tensor_float_out_npu(mean, std, generator, result);

  return result;
}

Tensor normal_float_tensor_npu(
    double mean, 
    const Tensor& std, 
    c10::optional<Generator> generator) {
  Tensor result = OpPreparation::ApplyTensor(std);
  normal_float_tensor_out_npu(mean, std, generator, result);

  return result;
}

Tensor normal_tensor_tensor_npu(
    const Tensor& mean, 
    const Tensor& std, 
    c10::optional<Generator> generator) {
  Tensor result = OpPreparation::ApplyTensor(mean);
  normal_tensor_tensor_out_npu(mean, std, generator, result);

  return result;
}

Tensor normal_float_float_npu(
    double mean, 
    double std, 
    IntArrayRef size,
    c10::optional<Generator> generator,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      size, dtype_opt, layout_opt, device_opt, pin_memory_opt, ACL_FORMAT_ND);

  // calculate the output result of the NPU
  normal_float_float_out_npu(mean, std, size, generator, result);

  return result;
}

Tensor& normal_npu_(
    Tensor& self,
    double mean,
    double std,
    c10::optional<Generator> generator) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = normal_float_float_out_npu(mean, std, contiguousSelf.sizes(), generator, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    normal_float_float_out_npu(mean, std, self.sizes(), generator, self);
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("normal_", TORCH_FN(normal_npu_));
  m.impl("normal.Tensor_float_out", TORCH_FN(normal_tensor_float_out_npu));
  m.impl("normal.Tensor_float", TORCH_FN(normal_tensor_float_npu));
  m.impl("normal.float_Tensor_out", TORCH_FN(normal_float_tensor_out_npu));
  m.impl("normal.float_Tensor", TORCH_FN(normal_float_tensor_npu));
  m.impl("normal.Tensor_Tensor_out", TORCH_FN(normal_tensor_tensor_out_npu));
  m.impl("normal.Tensor_Tensor", TORCH_FN(normal_tensor_tensor_npu));
  m.impl("normal.float_float", TORCH_FN(normal_float_float_npu));
  m.impl("normal.float_float_out", TORCH_FN(normal_float_float_out_npu));
}

} // namespace native
} // namespace at