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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& max_unpool2d_backward_out(
    const at::Tensor& gradOutput,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef outputSize,
    at::Tensor& gradInput) {
  OpPreparation::CheckOut(
      {self, gradOutput},
      gradInput,
      self); 
  TORCH_CHECK(
      outputSize.size() == 2,
      "There should be exactly two elements (height, width) in outputSize");
  TORCH_CHECK(
      (self.ndimension() == 3 || self.ndimension() == 4),
      "Input to max_unpooling2d should be a 3d or 4d Tensor");
  TORCH_CHECK(
      self.sizes() == indices.sizes(),
      "Shape of indices should match shape of input");
  TORCH_CHECK(self.numel() > 0, "Input must be non-empty");

  auto oheight = outputSize[0];
  auto owidth = outputSize[1];
  int64_t n = 1;
  int64_t c = self.size(0);
  int64_t h = self.size(1);
  int64_t w = self.size(2);
  int64_t selfDim = self.ndimension();

  if (selfDim == 4) {
    n = self.size(0);
    c = self.size(1);
    h = self.size(2);
    w = self.size(3);
  }

  auto gradOutputContiguous = gradOutput.contiguous();
  auto indicesContiguous = indices.contiguous();
  gradOutputContiguous = gradOutputContiguous.reshape({n, c, oheight * owidth});
  indicesContiguous = indicesContiguous.reshape({n, c, h * w});
  gradInput.resize_as_(self);
  gradInput.zero_();
  gradInput = gradInput.reshape({n, c, h * w});
  const int dim = 2;
  
  gradInput = NPUNativeFunctions::gather_out(gradOutputContiguous, dim, indicesContiguous, false, gradInput);
  if (selfDim == 3) {
    gradInput = gradInput.reshape({c, h, w});
  } else { 
    gradInput = gradInput.reshape({n, c, h, w});
  }
  return gradInput;
}

at::Tensor max_unpool2d_backward(
    const at::Tensor& gradOutput,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef outputSize) {
  auto gradInput = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  max_unpool2d_backward_out(gradOutput, self, indices, outputSize, gradInput);
  return gradInput;
}
} // namespace native
} // namespace at_npu
