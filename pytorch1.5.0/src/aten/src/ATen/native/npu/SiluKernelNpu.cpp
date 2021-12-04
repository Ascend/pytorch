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

Tensor& silu_out_npu_nocheck(Tensor& result, const Tensor& self) {
  OpCommand cmd;
  cmd.Name("Swish")
     .Input(self)
     .Output(result)
     .Run();
  return result;
}

Tensor& silu_out_npu(const Tensor& self, Tensor& out){
  OpPreparation::CheckOut(
      {self},
      out,
      self);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {out})
    .Func([&self](Tensor& out){silu_out_npu_nocheck(out, self);})
    .Call(out);
}

Tensor silu_npu(const Tensor& self) {
  OpPipeWithApplyOut pipe;
  return pipe.ApplyOutputSameAs(self)
    .Func([&self](Tensor& result) {silu_out_npu_nocheck(result, self);})
    .Call();
}

Tensor& silu_npu_(Tensor& self) {
  silu_out_npu(self, self);
  return self;
}

} // namespace native
} // namespace at
