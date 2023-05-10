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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

at::Tensor& logical_not_out_npu_nocheck(
    const at::Tensor& self, 
    at::Tensor& result) {
  at::ScalarType src_type = self.scalar_type();
  at::Tensor selfCast = self;
  if (src_type != at::ScalarType::Bool) {
    selfCast = NPUNativeFunctions::npu_dtype_cast(self, at::kBool);
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
  }
  OpCommand cmd;
  cmd.Name("LogicalNot")
      .Input(selfCast)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::logical_not_out(const at::Tensor& self, at::Tensor& result) {
  auto resultDtype = result.scalar_type();
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      resultDtype,
      self.sizes());
  OpPipeWithDefinedOut pipe;
  result = pipe.CheckMemory({self}, {result})
    .Func([&self](at::Tensor& result){logical_not_out_npu_nocheck(self, result);})
    .Call(result);
  result = NPUNativeFunctions::npu_dtype_cast(result, resultDtype);
  return result;
}

at::Tensor NPUNativeFunctions::logical_not(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(
      self.sizes(),
      self.options().dtype(at::kBool),
      self);
  logical_not_out_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::logical_not_(at::Tensor& self) {
  logical_not_out(self, self);
  return self;
}
} // namespace native
} // namespace at_npu
