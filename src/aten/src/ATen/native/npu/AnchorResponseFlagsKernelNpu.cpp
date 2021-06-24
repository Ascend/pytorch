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

static inline void anchor_response_flags_check(
    const Tensor& self,
    IntArrayRef featmap_size,
    IntArrayRef stride){
  TORCH_CHECK(
      featmap_size.size() == 2,
      "expected feat_map_size equals to 2, but got size ",
      featmap_size.size()); 
  TORCH_CHECK(
      self.dim() == 2 && self.size(1) == 4,
      "Non-empty 2D gt_bboxes tensor expected but got a tensor with sizes ",
      self.sizes());
  TORCH_CHECK(
      self.scalar_type() == ScalarType::Half || self.scalar_type() == ScalarType::Float,
      "float16 or float32 tensor expected but got a tensor with dtype: ",
      self.scalar_type());
}

Tensor anchor_response_flags_npu(
    const Tensor& self,
    IntArrayRef featmap_size,
    IntArrayRef stride,
    int64_t num_base_anchors){
  anchor_response_flags_check(self, featmap_size, stride);
  // calculate output size
  int64_t outputValue = featmap_size[0] * featmap_size[1] * num_base_anchors;
  SmallVector<int64_t, N> outputSize = {outputValue};
  auto options = self.options().dtype(ScalarType::Byte);
  Tensor result = OpPreparation::ApplyTensor(outputSize, options, self);
  Tensor selfCp = self.npu_dtype_cast(ScalarType::Float);

  OpCommand cmd;
  cmd.Name("AnchorResponseFlags")
      .Input(selfCp)
      .Output(result)
      .Attr("featmap_size", featmap_size)
      .Attr("strides", stride)
      .Attr("num_base_anchors", num_base_anchors)
      .Run();
  
  return result;
}

} // namespace native
} // namespace at
