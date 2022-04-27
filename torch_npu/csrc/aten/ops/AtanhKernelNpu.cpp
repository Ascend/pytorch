// Copyright (c) 2022 Huawei Technologies Co., Ltd
// Copyright (c) 2022, Facebook CORPORATION.
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

at::Tensor& atanh_out_npu_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Atanh")
     .Input(self)
     .Output(result)
     .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::atanh_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  if (!NpuUtils::check_match(&result)) {
    at::Tensor contiguousResult = NpuUtils::format_contiguous(result);
    atanh_out_npu_nocheck(self, contiguousResult);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    atanh_out_npu_nocheck(self, result);
  }
  return result;
}

at::Tensor NPUNativeFunctions::atanh(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  atanh_out_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::atanh_(at::Tensor& self) {
  return atanh_out(self, self);
}

}} // namespace at_npu::native