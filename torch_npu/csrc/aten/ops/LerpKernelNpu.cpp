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
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    lerp_out_npu_nocheck(self, end, weight, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
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
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    lerp_out_npu_nocheck(self, end, weight, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    lerp_out_npu_nocheck(self, end, weight, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::lerp(const at::Tensor& start, const at::Tensor& end, const at::Tensor& weight) {
  at::Tensor result = OpPreparation::ApplyTensor(start);
  lerp_out_npu_nocheck(start, end, weight, result);
  return result;
}

at::Tensor NPUNativeFunctions::lerp(const at::Tensor& start, const at::Tensor& end, at::Scalar weight) {
  at::Tensor result = OpPreparation::ApplyTensor(start);
  lerp_out_npu_nocheck(start, end, weight, result);
  return result;
}

at::Tensor& NPUNativeFunctions::lerp_(at::Tensor& self, const at::Tensor& end, const at::Tensor& weight) {
  OpPreparation::CheckMemory({self, end, weight}, {self});
  lerp_out(self, end, weight, self);
  return self;
}

at::Tensor& NPUNativeFunctions::lerp_(at::Tensor& self, const at::Tensor& end, at::Scalar weight) {
  OpPreparation::CheckMemory({self, end}, {self});
  lerp_out(self, end, weight, self);
  return self;
}
} // namespace native
} // namespace at_npu
