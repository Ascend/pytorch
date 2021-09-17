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

namespace {
SmallVector<int64_t, SIZE> diag_npu_output_size(
    const Tensor& self,
    int64_t diagonal) {
  SmallVector<int64_t, SIZE> shape;
  // input is 1-d
  if (self.dim() == 1) {
    shape.emplace_back(self.size(0) + diagonal);
    shape.emplace_back(self.size(0) + diagonal);
    return shape;
  }

  // input is 2-d
  int64_t m = self.size(0); // row
  int64_t n = self.size(1); // col
  if (m == n) {
    shape.emplace_back(m - diagonal);
  } else if (m < n) {
    shape.emplace_back(diagonal <= n - m ? m : n - diagonal);
  } else {
    shape.emplace_back(n - diagonal);
  }
  return shape;
}
} // namespace

Tensor& diag_out_npu_nocheck(Tensor& result, const Tensor& self, int64_t diagonal) {
  // judging and executing the NPU operator
  // If input is a 1-D tensor, then returns a 2-D square tensor with the elements of input as the diagonal.
  // If input is a matrix (2-D tensor), then returns a 1-D tensor with the diagonal elements of input.
  OpCommand cmd;
  if (self.dim() == 1) {
    cmd.Name("Diag");
  } else {
    cmd.Name("DiagPart");
  }
  cmd.Input(self)
    .Output(result)
    .Attr("diagonal", diagonal)
    .Run();

  return result;
}

Tensor& diag_out_npu(const Tensor& self, int64_t diagonal, Tensor& result) {
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2),
              "Value should be a 1-dimensional tensor or 2-dimensional tensor, but got ", self.dim());
  diagonal = make_wrap_dim(diagonal, self.dim());
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2 && diagonal <= self.size(0) && diagonal <= self.size(1)),
              "If the value is 2-dimensional tensor, the diagonal shoule less than shape.Diagonal is ", diagonal);

  auto outputSize = diag_npu_output_size(self, diagonal);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      self.scalar_type(),
      outputSize);

  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self, &diagonal](Tensor& result){diag_out_npu_nocheck(result, self, diagonal);})
   .Call(result);
}

Tensor diag_npu(const Tensor& self, int64_t diagonal) {
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2),
              "Value should be a 1-dimensional tensor or 2-dimensional tensor, but got ", self.dim());
  diagonal = make_wrap_dim(diagonal, self.dim());
  TORCH_CHECK((self.dim() == 1) || (self.dim() == 2 && diagonal <= self.size(0) && diagonal <= self.size(1)),
              "If the value is 2-dimensional tensor, the diagonal shoule less than shape.Diagonal is ", diagonal);

  // calculate the output size
  auto outputSize = diag_npu_output_size(self, diagonal);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  diag_out_npu_nocheck(result, self, diagonal);
  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("diag", TORCH_FN(diag_npu));
  m.impl("diag.out", TORCH_FN(diag_out_npu));
}
} // namespace native
} // namespace at