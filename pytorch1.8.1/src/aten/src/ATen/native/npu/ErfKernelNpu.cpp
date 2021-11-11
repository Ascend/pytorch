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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& erf_npu_nocheck(const Tensor& self, Tensor& out) {
  OpCommand cmd;
  cmd.Name("Erf")
    .Input(self)
    .Output(out)
    .Run();
  return out;
}

Tensor& erf_out_npu(const Tensor& self, Tensor& out) {
  OpPreparation::CheckOut(
      {self},
      out,
      self);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {out})
   .Func([&self](Tensor& out){erf_npu_nocheck(self, out);})
   .Call(out);
}

Tensor erf_npu(const Tensor& self) {
  auto outputSize = input_same_output_size(self); 
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  erf_npu_nocheck(self, result);
  return result;
}

Tensor& erf_npu_(Tensor& self) {
  erf_out_npu(self, self);
  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("erf_", TORCH_FN(erf_npu_));
  m.impl("erf", TORCH_FN(erf_npu));
  m.impl("erf.out", TORCH_FN(erf_out_npu));
}
} // namespace native
} // namespace at