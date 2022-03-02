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

at::Tensor& floor_out_npu_nocheck(const at::Tensor& self, at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Floor")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::floor_out(const at::Tensor& self, at::Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](at::Tensor& result){floor_out_npu_nocheck(self, result);})
   .Call(result);
}

at::Tensor& NPUNativeFunctions::floor_(at::Tensor& self) {
  NPUNativeFunctions::floor_out(self, self);

  return self;
}

at::Tensor NPUNativeFunctions::floor(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  floor_out_npu_nocheck(self, result);
  return result;
}

} // namespace native
} // namespace at_npu
