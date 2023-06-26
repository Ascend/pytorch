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
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

c10::SmallVector<int64_t, SIZE> lerp_broadcast_size(
    const at::Tensor& self, 
    const at::Tensor& end, 
    const at::Tensor& weight) {
  auto expanded_size = broadcast_ops_npu_output_size(self, end);
  auto output_size = broadcast_ops_npu_output_size(expanded_size, weight.sizes());
  return output_size;
}

at::Tensor& lerp_out_npu_nocheck(
    const at::Tensor& self, 
    const at::Tensor& end, 
    const at::Tensor& weight, 
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Lerp")
      .Input(self)
      .Input(end)
      .Input(weight)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& lerp_out_npu_nocheck(
    const at::Tensor& self, 
    const at::Tensor& end, 
    at::Scalar weight, 
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Lerp")
      .Input(self)
      .Input(end)
      .Input(weight, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::lerp_out(
    const at::Tensor& self, 
    const at::Tensor& end, 
    const at::Tensor& weight, 
    at::Tensor& result) {
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
      "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  auto output_size = lerp_broadcast_size(self, end, weight);
  OpPreparation::CheckOut(
      {self, end, weight},
      result,
      self,
      output_size);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    lerp_out_npu_nocheck(self, end, weight, contiguous_result);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    lerp_out_npu_nocheck(self, end, weight, result);
  }
  return result;
}

at::Tensor& NPUNativeFunctions::lerp_out(
    const at::Tensor& self, 
    const at::Tensor& end, 
    at::Scalar weight, 
    at::Tensor& result) {
  auto output_size = broadcast_ops_npu_output_size(self, end);
  OpPreparation::CheckOut(
      {self, end},
      result,
      self,
      output_size);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguous_result = NpuUtils::format_contiguous(result);
    lerp_out_npu_nocheck(self, end, weight, contiguous_result);
    NpuUtils::format_fresh_view(result, contiguous_result);
  } else {
    lerp_out_npu_nocheck(self, end, weight, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::lerp(const at::Tensor& self, const at::Tensor& end, const at::Tensor& weight) {
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
      "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  auto output_size = lerp_broadcast_size(self, end, weight);
  at::Tensor result = OpPreparation::ApplyTensor(self, output_size);
  lerp_out_npu_nocheck(self, end, weight, result);
  return result;
}

at::Tensor NPUNativeFunctions::lerp(const at::Tensor& self, const at::Tensor& end, at::Scalar weight) {
  auto output_size = broadcast_ops_npu_output_size(self, end);
  at::Tensor result = OpPreparation::ApplyTensor(self, output_size);
  lerp_out_npu_nocheck(self, end, weight, result);
  return result;
}

at::Tensor& NPUNativeFunctions::lerp_(at::Tensor& self, const at::Tensor& end, const at::Tensor& weight) {
  TORCH_CHECK(weight.dim() <= std::max(self.dim(), end.dim()),
      "weight should be of dimension max(self.dim(), end.dim()) or lesser");
  c10::SmallVector<int64_t, SIZE> self_size = array_to_small_vector(self.sizes());
  auto output_size = lerp_broadcast_size(self, end, weight);
  TORCH_CHECK(self_size == output_size,
      "output with shape ", self_size, " doesn't match the broadcast shape ", output_size);
  lerp_out(self, end, weight, self);
  return self;
}

at::Tensor& NPUNativeFunctions::lerp_(at::Tensor& self, const at::Tensor& end, at::Scalar weight) {
  c10::SmallVector<int64_t, SIZE> self_size = array_to_small_vector(self.sizes());
  auto output_size = broadcast_ops_npu_output_size(self, end);
  TORCH_CHECK(self_size == output_size,
      "output with shape ", self_size, " doesn't match the broadcast shape ", output_size);
  lerp_out(self, end, weight, self);
  return self;
}
} // namespace native
} // namespace at_npu
