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

at::Tensor& triu_out_npu_nocheck(const at::Tensor& self, int64_t k, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Triu")
    .Input(self)
    .Output(result)
    .Attr("diagonal", k)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::triu_out(const at::Tensor& self, int64_t k, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  at::Tensor selfCopy = self;
  if (self.scalar_type() == at::ScalarType::Half) {
    selfCopy = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
  triu_out_npu_nocheck(selfCopy, k, result);
  if (self.scalar_type() == at::ScalarType::Half) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Half);
  }
  return result;
}

at::Tensor NPUNativeFunctions::triu(const at::Tensor& self, int64_t k) {
  at::Tensor selfCopy = self;
  if (self.scalar_type() == at::ScalarType::Half) {
    selfCopy = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
  at::Tensor result = OpPreparation::ApplyTensor(selfCopy);

  triu_out_npu_nocheck(selfCopy, k, result);
  if (self.scalar_type() == at::ScalarType::Half) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Half);
  }

  return result;
}

at::Tensor& NPUNativeFunctions::triu_(at::Tensor& self, int64_t k) {
  at::Tensor selfCopy = self;
  if (self.scalar_type() == at::ScalarType::Half) {
    selfCopy = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(selfCopy);
    at::Tensor result = triu_out_npu_nocheck(contiguousSelf, k, contiguousSelf);
    if (self.scalar_type() == at::ScalarType::Half) {
      result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Half);
    }
    self.copy_(result);
  } else {
    triu_out_npu_nocheck(selfCopy, k, selfCopy);
    if (self.scalar_type() == at::ScalarType::Half) {
      selfCopy = NPUNativeFunctions::npu_dtype_cast(selfCopy, at::ScalarType::Half);
    }
    self.copy_(selfCopy);
  }

  return self;
}

} // namespace native
} // namespace at_npu
