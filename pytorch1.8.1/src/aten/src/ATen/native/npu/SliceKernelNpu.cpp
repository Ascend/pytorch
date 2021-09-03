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
#include <torch/script.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& slice_out_npu(
    const Tensor& self,
    IntArrayRef offsets,
    IntArrayRef size,
    Tensor& result) {

  SmallVector<int64_t, N> offsetVec = array_to_small_vector(offsets);
  SmallVector<int64_t, N> sizeVec = array_to_small_vector(size);
  OpCommand cmd;
  cmd.Name("Slice")
      .Input(self)
      .Input(offsetVec)
      .Input(sizeVec)
      .Output(result)
      .Run();
  return result;
}

Tensor slice_npu(const Tensor& self, IntArrayRef offsets, IntArrayRef size) {
  // calculate the output size
  SmallVector<int64_t, SIZE> outputSize = 
      CalcuOpUtil::ConvertIntArrayRefToSmallVector(size);
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  // calculate the output result of the NPU
  slice_out_npu(self, offsets, size, result);

  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_slice", TORCH_FN(slice_npu));
  m.impl("npu_slice.out", TORCH_FN(slice_out_npu));
}

Tensor npu_slice(const Tensor& self, IntArrayRef offsets, IntArrayRef size) {
  return at::native::slice_npu(self, offsets, size);
}

Tensor& npu_slice_out(const Tensor& self,
    IntArrayRef offsets,
    IntArrayRef size,
    Tensor& result) {
  return at::native::slice_out_npu(self, offsets, size, result);
}

} // namespace native
} // namespace at