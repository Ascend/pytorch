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

at::Tensor& erfc_out_npu_no_check(at::Tensor& out, const at::Tensor& self){
  OpCommand cmd;
  cmd.Name("Erfc")
    .Input(self)
    .Output(out)
    .Run();
  return out;
}

at::Tensor& NPUNativeFunctions::erfc_out(const at::Tensor& self, at::Tensor& out) {
  OpPreparation::CheckOut(
      {self},
      out,
      self,
      self.sizes());
  
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {out})
        .Func([&self](at::Tensor& out){erfc_out_npu_no_check(out, self);})
        .Call(out);
}

at::Tensor NPUNativeFunctions::erfc(const at::Tensor& self) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  erfc_out_npu_no_check(result, self);
  return result;
}

at::Tensor& NPUNativeFunctions::erfc_(at::Tensor& self) {
  NPUNativeFunctions::erfc_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu
