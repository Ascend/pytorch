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

std::tuple<Tensor&, Tensor&> giou_backward_inner_out_npu(
    Tensor& dbboxes,
    Tensor& dgtboxes,
    const Tensor& grad,
    const Tensor& bboxes,
    const Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode){
  string mode_str = mode == 1 ? "iof" : "iou";

  OpCommand cmd;
  cmd.Name("GIoUGrad")
      .Input(grad)
      .Input(bboxes)
      .Input(gtboxes)
      .Output(dbboxes)
      .Output(dgtboxes)
      .Attr("trans", trans)
      .Attr("is_cross", is_cross)
      .Attr("mode", mode_str)
      .Run();
  return std::tie(dbboxes, dgtboxes);
}

std::tuple<Tensor, Tensor> giou_backward_npu(
    const Tensor& grad,
    const Tensor& bboxes,
    const Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode){
  TORCH_CHECK(trans && !is_cross &&  mode == 0,
      "giou backward only support trans==True, ",
      "is_cross==False, ",
      "mode==0('iou') current version ",
      "if you need to back propagation, ",
      "please ensure your parameter is correct!");
  // Op need form of [n] grad
  // Note: temp avoid! it'll be remove while op deal with fp16 issue!
  Tensor gradCp = at::squeeze(grad, 0);
  if(gradCp.scalar_type() == at::kHalf){
    gradCp = gradCp.npu_dtype_cast(at::kFloat);
  }
  Tensor bboxesCp = bboxes;
  if(bboxesCp.scalar_type() == at::kHalf){
    bboxesCp = bboxesCp.npu_dtype_cast(at::kFloat);
  }
  Tensor gtboxesCp = gtboxes;
  if(gtboxesCp.scalar_type() == at::kHalf){
    gtboxesCp = gtboxesCp.npu_dtype_cast(at::kFloat);
  }
  Tensor dbboxes = OpPreparation::ApplyTensor(bboxesCp);
  Tensor dgtboxes = OpPreparation::ApplyTensor(gtboxesCp);

  giou_backward_inner_out_npu(dbboxes, dgtboxes, gradCp, bboxesCp, gtboxesCp, trans, is_cross, mode);
  if(bboxes.scalar_type() == at::kHalf || gtboxes.scalar_type() == at::kHalf){
    dbboxes = dbboxes.npu_dtype_cast(at::kHalf);
    dgtboxes = dgtboxes.npu_dtype_cast(at::kHalf);
  }
  return std::tie(dbboxes, dgtboxes);
}

} // namespace native
} // namespace at
