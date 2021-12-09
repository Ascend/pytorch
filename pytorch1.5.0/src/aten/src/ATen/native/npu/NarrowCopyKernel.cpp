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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor narrow_copy_npu(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t length) {
  int64_t dim_len = self.dim();

  TORCH_CHECK(dim_len > 0, "narrow() cannot be applied to a 0-dim tensor.");

  int64_t min = -dim_len;
  int64_t max = dim_len - 1;
  if (dim < min || dim > max) {
    AT_INDEX_ERROR(
        "Dimension out of range (expected to be in range of [",
        min, ", ", max, "], but got ", dim, ")");
  }
  if (dim < 0) {
    dim += dim_len;
  }

  auto cur_size = self.size(dim);
  if (start != cur_size) {  // start being the end is valid, but not a valid dim specification.
    start = maybe_wrap_dim(start, cur_size);
  }
  TORCH_CHECK(length >= 0 && start <= cur_size - length,
      "start (", start, ") + length (", length, ") exceeds dimension size (", cur_size, ").");

  SmallVector<int64_t, SIZE> outputSize;
  outputSize = input_same_output_size(self);
  outputSize[dim] = length;

  Tensor result = OpPreparation::ApplyTensor(self, outputSize);

  SmallVector<int64_t, N> offsetList(self.dim(), 0);
  offsetList[dim] = start;

  SmallVector<int64_t, N> sizeList(self.dim(), -1);
  sizeList[dim] = length;

  OpCommand cmd;
  cmd.Name("Slice")
      .Input(self)
      .Input(offsetList)
      .Input(sizeList)
      .Output(result)
      .Run();
  return result.clone(at::MemoryFormat::Contiguous);
}

} // namespace native
} // namespace at
