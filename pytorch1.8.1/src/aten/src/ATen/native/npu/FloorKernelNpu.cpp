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

Tensor& floor_out_npu_nocheck(const Tensor& self, Tensor& result) {
  OpCommand cmd;
  cmd.Name("Floor")
      .Input(self)
      .Output(result)
      .Run();
      
  return result;
}

Tensor& floor_out_npu(const Tensor& self, Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      self.scalar_type(),
      self.sizes());

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](Tensor& result){floor_out_npu_nocheck(self, result);})
   .Call(result);
}

Tensor& floor_npu_(Tensor& self) {
  floor_out_npu(self, self);

  return self;
}

Tensor floor_npu(const Tensor& self) {
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      self.sizes(), self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  floor_out_npu_nocheck(self, result);
  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("floor", TORCH_FN(floor_npu));
  m.impl("floor_", TORCH_FN(floor_npu_));
  m.impl("floor.out", TORCH_FN(floor_out_npu));
}

} // namespace native
} // namespace at
