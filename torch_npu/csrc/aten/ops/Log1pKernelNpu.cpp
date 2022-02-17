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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/OpTemplate.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu { 
namespace native {

at::Tensor& log1p_out_npu_nocheck(const at::Tensor& self, at::Tensor& result){   
  OpCommand cmd;
  cmd.Name("Log1p")
      .Input(self)
      .Output(result)
      .Run();
  return result; 
}

at::Tensor& NPUNativeFunctions::log1p_out(const at::Tensor& self, at::Tensor& result){ 
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](at::Tensor& result){log1p_out_npu_nocheck(self, result);})
   .Call(result);
}

at::Tensor NPUNativeFunctions::log1p(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  log1p_out_npu_nocheck(self, result);
  return result;
}

at::Tensor& NPUNativeFunctions::log1p_(at::Tensor& self) {
  log1p_out(self, self);
  return self;
}
}}
