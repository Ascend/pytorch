// Copyright (c) 2020, Huawei Technologies.
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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& nonzero_out_npu_nocheck(Tensor& result, const Tensor& self) {
  OpCommand cmd;
  cmd.Name("NonZero")
    .Input(self)
    .Output(result)
    .Attr("transpose", false)
    .Run();

  return result;
}

Tensor& nonzero_out_npu(Tensor& result, const Tensor& self) {
  auto outputSize = nonzero_npu_output_size(self);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      ScalarType::Long,
      outputSize);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self](Tensor& result){nonzero_out_npu_nocheck(result, self);})
   .Call(result);
}

Tensor nonzero_npu(const Tensor& self) {
  Tensor dtypeCastOfSelf = self;

  // calculate the output size
  auto outputSize = nonzero_npu_output_size(dtypeCastOfSelf);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, dtypeCastOfSelf.options().dtype(kLong), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  nonzero_out_npu_nocheck(result, dtypeCastOfSelf);
  return result;
}

} // namespace native
} // namespace at
