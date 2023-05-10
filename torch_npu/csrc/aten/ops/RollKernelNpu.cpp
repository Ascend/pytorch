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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu { 
namespace native {

at::Tensor& roll_out_npu_no_transpose(
    at::Tensor& result, 
    const at::Tensor& self, 
    at::IntArrayRef shifts, 
    at::IntArrayRef dims) {
  
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

at::Tensor& roll_transpose(
    at::Tensor& result, 
    const at::Tensor& self, 
    int64_t axis, 
    int64_t firstDim, 
    at::IntArrayRef shifts, 
    int64_t id) {

  c10::SmallVector<int64_t, SHAPE_SIZE> perm;
  for (int64_t i = 0; i < self.dim(); i++) {
    perm.emplace_back(i);
  }
  std::swap(perm[axis], perm[firstDim]);
  at::Tensor transposeSelf = NPUNativeFunctions::npu_transpose(self, perm, true);
  auto outputSize = transpose_npu_output_size(result, perm);
  at::Tensor transposeResult = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      self.options(), CalcuOpUtil::GetTensorNpuFormat(self));
  c10::SmallVector<int64_t, SIZE> dim = {firstDim};
  c10::SmallVector<int64_t, SIZE> shift_bak = {shifts[id]};
  at::IntArrayRef dim_now = at::IntArrayRef(dim);
  at::IntArrayRef shift_now = at::IntArrayRef(shift_bak);
  roll_out_npu_no_transpose(transposeResult, transposeSelf, shift_now, dim_now);
  NPUNativeFunctions::npu_transpose_out(transposeResult, perm, true, result);
  return result;
}

at::Tensor& roll_out_npu(
    at::Tensor& result, 
    const at::Tensor& self, 
    at::IntArrayRef shifts, 
    at::IntArrayRef dims) {
  
  if (dims.size() == 0) {
    roll_out_npu_no_transpose(result, self, shifts, dims);
  } else {
    TORCH_CHECK(dims.size() == shifts.size(), 
                "The size of shifts and dims should be the same when the size of dims is not 0.");
    int64_t firstDim = CalcuOpUtil::MakeWrapDim(0, self.dim());
    for (int i = 0; i < dims.size(); i++) {
      int64_t axis = CalcuOpUtil::MakeWrapDim(dims[i], self.dim());
      if (i == 0) {
        if (axis == firstDim) {
          c10::SmallVector<int64_t, SIZE> dim = {firstDim};
          c10::SmallVector<int64_t, SIZE> shift_bak = {shifts[i]};
          at::IntArrayRef dim_now = at::IntArrayRef(dim);
          at::IntArrayRef shift_now = at::IntArrayRef(shift_bak);
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

at::Tensor NPUNativeFunctions::roll(
    const at::Tensor& self, 
    at::IntArrayRef shifts, 
    at::IntArrayRef dims) {
    
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self);
    
  // calculate the output result of the NPU
  roll_out_npu(result, self, shifts, dims);
  return result;
}
} // namespace native
} // namespace at_npu