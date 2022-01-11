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

Tensor& max_unpool2d_out_npu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef outputSize,
    Tensor& output) {
  OpPreparation::CheckOut(
      {self, indices},
      output,
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
  auto selfContiguous = self.contiguous();
  auto indicesContiguous = indices.contiguous();
  int64_t h = -1;
  int64_t w = -1;
  int64_t selfDim = self.ndimension();
  int64_t numBatch = -1;
  int64_t numChannels = -1;

  if (selfDim == 3) {
    numChannels = self.size(0);
    h = self.size(1);
    w = self.size(2);
    output.resize_({numChannels, oheight * owidth});
    selfContiguous = selfContiguous.reshape({numChannels, h * w});
    indicesContiguous = indicesContiguous.reshape({numChannels, h * w});
  } else {
    numBatch = self.size(0);
    numChannels = self.size(1);
    h = self.size(2);
    w = self.size(3);
    output.resize_({numBatch, numChannels, oheight * owidth});
    selfContiguous = selfContiguous.reshape({numBatch, numChannels, h * w});
    indicesContiguous = indicesContiguous.reshape({numBatch, numChannels, h * w});
  }

  output.zero_();
  int64_t dim = 2;
  output = output.scatter(dim, indicesContiguous, selfContiguous);

  if (selfDim == 3) {
    output = output.reshape({numChannels, oheight, owidth});
  } else {
    output = output.reshape({numBatch, numChannels, oheight, owidth});
  }
  return output;
};

Tensor max_unpool2d_npu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size) {
  auto output = OpPreparation::ApplyTensor(self, {0});
  max_unpool2d_out_npu(self, indices, output_size, output);
  return output;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("max_unpool2d", TORCH_FN(max_unpool2d_npu));
  m.impl("max_unpool2d.out", TORCH_FN(max_unpool2d_out_npu));
}
} // namespace native
} // namespace at
