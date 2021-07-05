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
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/common/InnerNpuNativeFunction.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& mv_out_npu_nocheck(Tensor& result, const Tensor& self, const Tensor& vec) {
  bool isSelfT = CalcuOpUtil::is_transpose_last_two_dims(self);
  Tensor contiguousSelf;
  contiguousSelf = isSelfT ? self : NpuUtils::format_contiguous(self);
  Tensor vecT = at::unsqueeze(vec, 1);

  OpCommand cmd;
  cmd.Name("MatMul")
      .Input(contiguousSelf)
      .Input(vecT)
      .Attr("transpose_x1", isSelfT)
      .Attr("transpose_x2", false)
      .Output(result)
      .Run();

  result = at::squeeze(result, 1);
  npu_fast_reshape_(result);
  return result;
}

Tensor& mv_out_npu(Tensor& result, const Tensor& self, const Tensor& vec) {

  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      self.scalar_type(),
      {self.size(0)});

  result = at::unsqueeze(result, 1);
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self, vec}, {result})
      .Func([&self, &vec](Tensor& result){mv_out_npu_nocheck(result, self, vec);})
      .Call(result);
}

Tensor mv_npu(const Tensor& self, const Tensor& vec) {

  Tensor result = OpPreparation::ApplyTensor(self, {self.size(0), 1});

  // calculate the output result of the NPU
  mv_out_npu_nocheck(result, self, vec);

  return result;
}

} // namespace native
} // namespace at
