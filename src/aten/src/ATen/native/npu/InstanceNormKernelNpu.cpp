// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include <c10/npu/NPUCachingAllocator.h>
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<NPUTensorDesc, N> instance_norm_npu_input(const SmallVector<Tensor, N>& inputTensor) {
  return CalcuOpUtil::create_npu_input_tensor_desc(inputTensor);
}

SmallVector<NPUTensorDesc, N> instance_norm_npu_output(const Tensor& result) {
  return CalcuOpUtil::create_npu_output_tensor_desc({result});
}

SmallVector<NPUAttrDesc, N> instance_norm_npu_attr(bool use_input_stats, double momentum, double eps) {
  NPUAttrDesc npuAttrStats = NPUAttrDesc("use_input_stats", use_input_stats);
  NPUAttrDesc npuAttrMomentum = NPUAttrDesc("momentum", static_cast<float>(momentum));
  NPUAttrDesc npuAttrEpsilon = NPUAttrDesc("eps", static_cast<float>(eps));
  SmallVector<NPUAttrDesc, N> attrs = {npuAttrStats, npuAttrMomentum, npuAttrEpsilon};
  return attrs;
}

Tensor& instance_norm_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool use_input_stats,
    double momentum,
    double eps) {
  // constructs the input and output NPUTensorDesc
  auto inputs = instance_norm_npu_input(
      {self, weight, bias, running_mean, running_var});
  auto outputs = instance_norm_npu_output(result);

  // constructs the attr of the NPUAttrDesc
  auto attrs = instance_norm_npu_attr(use_input_stats, momentum, eps);

  // executing the NPU operator
  CalcuOpUtil::execute_npu_operate("InstanceNorm", inputs, outputs, attrs);

  return result;
}

Tensor instance_norm_npu(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool use_input_stats,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  TORCH_CHECK(use_input_stats || (running_mean.defined() && running_var.defined()), 
              "Expected running_mean and running_var to be defined when use_input_stats is false");
  Tensor result = at::empty_with_format(self.sizes(), self.options(), 
                                        CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  instance_norm_out_npu(
      result,
      self,
      weight,
      bias,
      running_mean,
      running_var,
      use_input_stats,
      momentum,
      eps);

  return result;
}

} // namespace native
} // namespace at
