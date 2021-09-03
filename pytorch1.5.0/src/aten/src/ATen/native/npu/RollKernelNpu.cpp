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

Tensor& roll_out_npu_no_transpose(Tensor& result, const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  // executing the NPU operator  
  OpCommand cmd;
  cmd.Name("Roll")
      .Input(self)
      .Output(result)
      .Attr("shifts", shifts)
      .Attr("dims", dims)
      .Run();

  return result;
}

Tensor& roll_transpose(Tensor& result, const Tensor& self, int64_t axis, int64_t firstDim, IntArrayRef shifts, int64_t id) {
  SmallVector<int64_t, SHAPE_SIZE> perm;
  for (int64_t i = 0; i < self.dim(); i++) {
    perm.emplace_back(i);
  }
  std::swap(perm[axis], perm[firstDim]);
  Tensor transposeSelf = at::npu_transpose(self, perm);
  auto outputSize = transpose_npu_output_size(result, perm);
  Tensor transposeResult = at::empty_with_format(
    outputSize,
    self.options(),
    CalcuOpUtil::get_tensor_npu_format(self));
  SmallVector<int64_t, SIZE> dim = {firstDim};
  SmallVector<int64_t, SIZE> shift_bak = {shifts[id]};
  IntArrayRef dim_now = IntArrayRef(dim);
  IntArrayRef shift_now = IntArrayRef(shift_bak);
  roll_out_npu_no_transpose(transposeResult, transposeSelf, shift_now, dim_now);
  at::npu_transpose_out(result, transposeResult, perm);
  return result;
}

Tensor& roll_out_npu(Tensor& result, const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  if (dims.size() == 0) {
    roll_out_npu_no_transpose(result, self, shifts, dims);
  } else {
    TORCH_CHECK(dims.size() == shifts.size(), 
                "The size of shifts and dims should be the same when the size of dims is not 0.");
    int64_t firstDim = CalcuOpUtil::make_wrap_dim(0, self.dim());
    for (int i = 0; i < dims.size(); i++) {
      int64_t axis = CalcuOpUtil::make_wrap_dim(dims[i], self.dim());
      if (i == 0) {
        if (axis == firstDim) {
          SmallVector<int64_t, SIZE> dim = {firstDim};
          SmallVector<int64_t, SIZE> shift_bak = {shifts[i]};
          IntArrayRef dim_now = IntArrayRef(dim);
          IntArrayRef shift_now = IntArrayRef(shift_bak);
          roll_out_npu_no_transpose(result, self, shift_now, dim_now);
        } else {
          roll_transpose(result, self, axis, firstDim, shifts, i);
        }
      } else {
        roll_transpose(result, result, axis, firstDim, shifts, i);
      }
    }
  }
  return result;
}

Tensor roll_npu(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self);
    
  // calculate the output result of the NPU
  roll_out_npu(result, self, shifts, dims);
  return result;
}

}
}