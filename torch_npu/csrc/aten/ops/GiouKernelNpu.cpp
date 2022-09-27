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

#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

c10::SmallVector<int64_t, N> giou_output_size(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool is_cross){
  c10::SmallVector<int64_t, N> output_size;
  if(is_cross){
      output_size = {gtboxes.size(0), self.size(0)};
  } else {
      output_size = {1, self.size(0)};
  }
  return output_size;
}

at::Tensor& giou_inner_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode){
  auto output_size = giou_output_size(self, gtboxes, is_cross);
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

at::Tensor giou_npu(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
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
  at::Tensor selfCp = self.permute({1, 0});
  if (selfCp.scalar_type() == at::kHalf) {
    selfCp = NPUNativeFunctions::npu_dtype_cast(selfCp, at::kFloat);
  }
  at::Tensor gtboxesCp = gtboxes.permute({1, 0});
  if (gtboxesCp.scalar_type() == at::kHalf) {
    gtboxesCp = NPUNativeFunctions::npu_dtype_cast(gtboxesCp, at::kFloat);
  }
  auto output_size = giou_output_size(selfCp, gtboxesCp, is_cross);
  at::Tensor result = OpPreparation::ApplyTensor(selfCp, output_size);

  giou_inner_out_npu(result, selfCp, gtboxesCp, trans, is_cross, mode);
  result = result.permute({1, 0});
  if (self.scalar_type() == at::kHalf || gtboxes.scalar_type() == at::kHalf) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kHalf);
  }
  return result;
}

std::tuple<at::Tensor&, at::Tensor&> giou_backward_inner_out_npu(
    at::Tensor& dbboxes,
    at::Tensor& dgtboxes,
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
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

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_giou_backward(
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
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
  at::Tensor gradCp = at::squeeze(grad, 0);
  if (gradCp.scalar_type() == at::kHalf) {
    gradCp = NPUNativeFunctions::npu_dtype_cast(gradCp, at::kFloat);
  }
  at::Tensor bboxesCp = bboxes;
  if (bboxesCp.scalar_type() == at::kHalf) {
    bboxesCp = NPUNativeFunctions::npu_dtype_cast(bboxesCp, at::kFloat);
  }
  at::Tensor gtboxesCp = gtboxes;
  if (gtboxesCp.scalar_type() == at::kHalf) {
    gtboxesCp = NPUNativeFunctions::npu_dtype_cast(gtboxesCp, at::kFloat);
  }
  at::Tensor dbboxes = OpPreparation::ApplyTensor(bboxesCp);
  at::Tensor dgtboxes = OpPreparation::ApplyTensor(gtboxesCp);

  giou_backward_inner_out_npu(dbboxes, dgtboxes, gradCp, bboxesCp, gtboxesCp, trans, is_cross, mode);
  if (bboxes.scalar_type() == at::kHalf || gtboxes.scalar_type() == at::kHalf) {
    dbboxes = NPUNativeFunctions::npu_dtype_cast(dbboxes, at::kHalf);
    dgtboxes = NPUNativeFunctions::npu_dtype_cast(dgtboxes, at::kHalf);
  }
  return std::tie(dbboxes, dgtboxes);
}

class NPUGiouFunction : public torch::autograd::Function<NPUGiouFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode) {
    ctx->saved_data["trans"] = trans;
    ctx->saved_data["is_cross"] = is_cross;
    ctx->saved_data["mode"] = mode;
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self, gtboxes});
    return giou_npu(self, gtboxes, trans, is_cross, mode);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto trans = ctx->saved_data["trans"].toBool();
    auto is_cross = ctx->saved_data["is_cross"].toBool();
    auto mode = ctx->saved_data["mode"].toInt();
    auto saved = ctx->get_saved_variables();
    auto bboxes = saved[0];
    auto gtboxes = saved[1];

    tuple<at::Tensor, at::Tensor> result = NPUNativeFunctions::npu_giou_backward(grad_outputs[0],
        bboxes, gtboxes, trans, is_cross, mode);

    tensor_list output = {std::get<0>(result),
        std::get<1>(result),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
    return output;
  }
};

at::Tensor NPUNativeFunctions::npu_giou(const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode) {
  return NPUGiouFunction::apply(self, gtboxes, trans, is_cross, mode);
}

} // namespace native
} // namespace at