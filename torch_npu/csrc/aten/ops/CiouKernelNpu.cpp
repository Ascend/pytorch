// Copyright (c) 2022 Huawei Technologies Co., Ltd
// Copyright (c) 2012, Facebook CORPORATION.
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

c10::SmallVector<int64_t, N> ciou_output_size(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool is_cross){
  c10::SmallVector<int64_t, N> output_size;
  if(is_cross){
      output_size = {gtboxes.size(1), self.size(1)};
  } else {
      output_size = {1, self.size(1)};
  }
  return output_size;
}

tuple<at::Tensor, at::Tensor> ciou_inner_out_npu(
    at::Tensor& overlap,
    at::Tensor& atan_sub,
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag){
  string mode_str = mode == 1 ? "iof" : "iou";
  OpCommand cmd;
  cmd.Name("CIoU")
      .Input(self)
      .Input(gtboxes)
      .Output(overlap)
      .Output(atan_sub)
      .Attr("trans", trans)
      .Attr("is_cross", is_cross)
      .Attr("mode", mode_str)
      .Attr("atan_sub_flag", atan_sub_flag)
      .Run();
  return std::tie(overlap, atan_sub);
}

tuple<at::Tensor, at::Tensor> ciou_npu(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag) {
  at::Tensor selfCp = self;
  if (selfCp.scalar_type() == at::kHalf) {
    selfCp = NPUNativeFunctions::npu_dtype_cast(selfCp, at::kFloat);
  }
  at::Tensor gtboxesCp = gtboxes;
  if (gtboxesCp.scalar_type() == at::kHalf) {
    gtboxesCp = NPUNativeFunctions::npu_dtype_cast(gtboxesCp, at::kFloat);
  }
  auto output_size = ciou_output_size(selfCp, gtboxesCp, is_cross);
  at::Tensor overlap = OpPreparation::ApplyTensor(selfCp, output_size);
  at::Tensor atan_sub = OpPreparation::ApplyTensor(selfCp, output_size);
  ciou_inner_out_npu(overlap, atan_sub, selfCp, gtboxesCp, trans, is_cross, mode, atan_sub_flag);
  if (self.scalar_type() == at::kHalf || gtboxes.scalar_type() == at::kHalf) {
    overlap = NPUNativeFunctions::npu_dtype_cast(overlap, at::kHalf);
  }
  return std::tie(overlap, atan_sub);
}

std::tuple<at::Tensor&, at::Tensor&> ciou_backward_inner_out_npu(
    at::Tensor& dbboxes,
    at::Tensor& dgtboxes,
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    const at::Tensor& atan_sub,
    bool trans,
    bool is_cross,
    int64_t mode){
  string mode_str = mode == 1 ? "iof" : "iou";
  OpCommand cmd;
  cmd.Name("CIoUGrad")
      .Input(grad)
      .Input(bboxes)
      .Input(gtboxes)
      .Input(atan_sub)
      .Output(dbboxes)
      .Output(dgtboxes)
      .Attr("trans", trans)
      .Attr("is_cross", is_cross)
      .Attr("mode", mode_str)
      .Run();

  return std::tie(dbboxes, dgtboxes);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::npu_ciou_backward(
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    const c10::optional<at::Tensor>& atan_sub_opt,
    bool trans,
    bool is_cross,
    int64_t mode){
  const at::Tensor& atan_sub = c10::value_or_else(atan_sub_opt, [] {return at::Tensor();});
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

  ciou_backward_inner_out_npu(dbboxes, dgtboxes, gradCp, bboxesCp, gtboxesCp, atan_sub, trans, is_cross, mode);

  if (bboxes.scalar_type() == at::kHalf || gtboxes.scalar_type() == at::kHalf) {
    dbboxes = NPUNativeFunctions::npu_dtype_cast(dbboxes, at::kHalf);
    dgtboxes = NPUNativeFunctions::npu_dtype_cast(dgtboxes, at::kHalf);
  }
  return std::tie(dbboxes, dgtboxes);
}

class NPUCiouFunction : public torch::autograd::Function<NPUCiouFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag) {
    ctx->saved_data["trans"] = trans;
    ctx->saved_data["is_cross"] = is_cross;
    ctx->saved_data["mode"] = mode;
    at::AutoNonVariableTypeMode g;
    auto result = ciou_npu(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
    ctx->save_for_backward({self, gtboxes, std::get<1>(result)});
    return std::get<0>(result);
  }

  static tensor_list backward(AutogradContext *ctx,
    tensor_list grad_outputs) {
    auto trans = ctx->saved_data["trans"].toBool();
    auto is_cross = ctx->saved_data["is_cross"].toBool();
    auto mode = ctx->saved_data["mode"].toInt();
    auto saved = ctx->get_saved_variables();
    auto bboxes = saved[0];
    auto gtboxes = saved[1];
    auto atan_sub = saved[2];

    tuple<at::Tensor, at::Tensor> result = NPUNativeFunctions::npu_ciou_backward(grad_outputs[0],
        bboxes, gtboxes, atan_sub, trans, is_cross, mode);

    tensor_list output = {std::get<0>(result),
        std::get<1>(result),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
    return output;
  }
};

at::Tensor NPUNativeFunctions::npu_ciou(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag) {
  return NPUCiouFunction::apply(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}
} // namespace native
} // namespace at