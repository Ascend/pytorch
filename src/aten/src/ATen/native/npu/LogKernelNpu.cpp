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

Tensor& log_out_npu_nocheck(Tensor& result, const Tensor& self) {
  OpCommand cmd;
  cmd.Name("Log")
      .Input(self)
      .Output(result)
      .Attr("base", (float)-1)
      .Attr("scale", (float)1)
      .Attr("shift", (float)0)
      .Run();

  return result;
}

Tensor& log_out_npu(Tensor& result, const Tensor& self) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](Tensor& result){log_out_npu_nocheck(result, self);})
   .Call(result);
}

Tensor log_npu(const Tensor& self) {
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  log_out_npu_nocheck(result, self);

  return result;
}

Tensor& log_npu_(Tensor& self) {
  log_out_npu(self, self);

  return self;
}

} // namespace native
} // namespace at