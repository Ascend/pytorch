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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& erfc_out_npu_no_check(Tensor& out, const Tensor& self){
  OpCommand cmd;
  cmd.Name("Erfc")
    .Input(self)
    .Output(out)
    .Run();
  return out;
}

Tensor& erfc_out_npu(Tensor& out, const Tensor& self) {
  OpPreparation::CheckOut(
      {self},
      out,
      self,
      self.sizes());
  
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {out})
        .Func([&self](Tensor& out){erfc_out_npu_no_check(out, self);})
        .Call(out);
}

Tensor erfc_npu(const Tensor& self) {
  Tensor result = OpPreparation::ApplyTensor(self);
  erfc_out_npu_no_check(result, self);
  return result;
}

Tensor& erfc_npu_(Tensor& self) {
  erfc_out_npu(self, self);
  return self;
}

} // native
} // at
