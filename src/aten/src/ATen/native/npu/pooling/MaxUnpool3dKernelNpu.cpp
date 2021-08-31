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

namespace {

SmallVector<int64_t, SIZE> max_pool3d_npu_output_size(
    const Tensor& self, 
    IntArrayRef output_size) {
  SmallVector<int64_t, SIZE> shape = {};
  if (self.dim() == 4) {
    shape = {self.size(0), output_size[0], output_size[1], output_size[2]};
  } else {
    shape = {self.size(0), self.size(1), output_size[0], output_size[1], output_size[2]};
  }
  return shape;
}  // max_pool3d_npu_output_size

}  // namespace

Tensor& max_unpool3d_out_npu_nocheck(
    Tensor& result, 
    const Tensor& self, 
    const Tensor& indices, 
    const Tensor& data,
    IntArrayRef output_size) {
  int64_t N = 1;
  int64_t C = self.size(0);
  if (self.dim() == 5) {
    N = self.size(0);
    C = self.size(1);
  }
  Tensor reshape_self = self.reshape({N, C, -1});
  Tensor reshape_indices = indices.reshape({N, C, -1});
  Tensor reshape_data = data.reshape({N, C, -1});
  result = result.reshape({N, C, -1});

  int64_t axis = 2;
  OpCommand cmd;
  cmd.Name("ScatterElements")
     .Input(reshape_data)
     .Input(reshape_indices)
     .Input(reshape_self)
     .Output(result)
     .Attr("axis", axis)
     .Run();
  result = result.reshape({data.sizes()});
  return result;
}

Tensor& max_unpool3d_out_npu(
    Tensor& result, 
    const Tensor& self, 
    const Tensor& indices, 
    IntArrayRef output_size, 
    IntArrayRef stride, 
    IntArrayRef padding) {
  auto out_shape = max_pool3d_npu_output_size(self, output_size);

  Tensor data = zeros_npu(out_shape, self.options());
  OpPreparation::CheckOut(
    {self, indices, data},
    result,
    data);

  max_unpool3d_out_npu_nocheck(result, self, indices, data, output_size);

  return result;
}

Tensor max_unpool3d_npu(
    const Tensor& self, 
    const Tensor& indices, 
    IntArrayRef output_size, 
    IntArrayRef stride, 
    IntArrayRef padding) {
  TORCH_CHECK(
      output_size.size() == 3,
      "There should be exactly 3 elements (depth, height, width) in output_size");
  TORCH_CHECK(
      (self.ndimension() == 4 || self.ndimension() == 5),
      "Input to max_unpooling2d should be a 4d or 5d Tensor");
  TORCH_CHECK(
      self.sizes() == indices.sizes(),
      "Shape of indices should match shape of input");
  TORCH_CHECK(self.numel() > 0, "Input must be non-empty");

  auto out_shape = max_pool3d_npu_output_size(self, output_size);

  Tensor data = zeros_npu(out_shape, self.options());
  Tensor result = OpPreparation::ApplyTensor(data);

  max_unpool3d_out_npu_nocheck(result, self, indices, data, output_size);

  return result;
}

} // namespace native
} // namespace at
