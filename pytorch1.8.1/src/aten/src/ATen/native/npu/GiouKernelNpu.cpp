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

SmallVector<int64_t, N> giou_output_size(
    const Tensor& self,
    const Tensor& gtboxes,
    bool is_cross){
  SmallVector<int64_t, N> output_size;
  if(is_cross){
      output_size = {gtboxes.size(0), self.size(0)};
  } else {
      output_size = {1, self.size(0)};
  }
  return output_size;
}

Tensor& giou_inner_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode){
  auto output_size = giou_output_size(self, gtboxes, is_cross);
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      output_size);
  string mode_str = mode == 1 ? "iof" : "iou";

  OpCommand cmd;
  cmd.Name("GIoU")
      .Input(self)
      .Input(gtboxes)
      .Output(result)
      .Attr("trans", trans)
      .Attr("is_cross", is_cross)
      .Attr("mode", mode_str)
      .Run();
  return result;
}

Tensor giou_npu(
    const Tensor& self,
    const Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode) {
  TORCH_CHECK(trans && !is_cross &&  mode == 0,
      "giou backward only support trans==True, ",
      "is_cross==False, ",
      "mode==0('iou') current version ",
      "if you need to back propagation, ",
      "please ensure your parameter is correct!");
  // Op need form of [n, 4], but pass should be [4, n];
  // Note: temp avoid! it'll be removed while op deal with fp16 issue!
  Tensor selfCp = self.permute({1, 0});
  if (selfCp.scalar_type() == at::kHalf) {
    selfCp = selfCp.npu_dtype_cast(at::kFloat);
  }
  Tensor gtboxesCp = gtboxes.permute({1, 0});
  if (gtboxesCp.scalar_type() == at::kHalf) {
    gtboxesCp = gtboxesCp.npu_dtype_cast(at::kFloat);
  }
  auto output_size = giou_output_size(selfCp, gtboxesCp, is_cross);
  Tensor result = OpPreparation::ApplyTensor(selfCp, output_size);

  giou_inner_out_npu(result, selfCp, gtboxesCp, trans, is_cross, mode);
  result = result.permute({1, 0});
  if (self.scalar_type() == at::kHalf || gtboxes.scalar_type() == at::kHalf) {
    result = result.npu_dtype_cast(at::kHalf);
  }
  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m){
  m.impl("npu_giou", TORCH_FN(giou_npu));
}

} // namespace native
} // namespace at
