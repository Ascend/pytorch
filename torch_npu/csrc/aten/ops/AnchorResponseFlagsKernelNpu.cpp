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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
static inline void anchor_response_flags_check(
    const at::Tensor& self,
    at::IntArrayRef featmap_size,
    at::IntArrayRef stride){
  TORCH_CHECK(
      featmap_size.size() == 2,
      "expected feat_map_size equals to 2, but got size ",
      featmap_size.size()); 
  TORCH_CHECK(
      self.dim() == 2 && self.size(1) == 4,
      "Non-empty 2D gt_bboxes tensor expected but got a tensor with sizes ",
      self.sizes());
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::Float,
      "float16 or float32 tensor expected but got a tensor with dtype: ",
      self.scalar_type());
}

at::Tensor NPUNativeFunctions::npu_anchor_response_flags(
    const at::Tensor& self,
    at::IntArrayRef featmap_size,
    at::IntArrayRef stride,
    int64_t num_base_anchors){
  anchor_response_flags_check(self, featmap_size, stride);
  // calculate output size
  int64_t outputValue = featmap_size[0] * featmap_size[1] * num_base_anchors;
  c10::SmallVector<int64_t, N> outputSize = {outputValue};
  auto options = self.options().dtype(at::ScalarType::Byte);
  at::Tensor result = OpPreparation::ApplyTensor(outputSize, options, self);
  NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);

  OpCommand cmd;
  cmd.Name("AnchorResponseFlags")
      .Input(self)
      .Output(result)
      .Attr("featmap_size", featmap_size)
      .Attr("strides", stride)
      .Attr("num_base_anchors", num_base_anchors)
      .Run();
  
  return result;
}

} // namespace native
} // namespace at
