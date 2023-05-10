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

at::Tensor NPUNativeFunctions::nll_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  return std::get<0>(
      at::nll_loss_forward(self, target, weight, reduction, ignore_index));
}

at::Tensor& NPUNativeFunctions::nll_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& output) {
  at::Tensor total_weight = OpPreparation::ApplyTensor({}, self.options(), self);
  return std::get<0>(at::nll_loss_forward_out(
      output, total_weight, self, target, weight, reduction, ignore_index));
}

at::Tensor NPUNativeFunctions::nll_loss2d(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  return std::get<0>(
      at::nll_loss2d_forward(self, target, weight, reduction, ignore_index));
}

at::Tensor& NPUNativeFunctions::nll_loss2d_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& output) {
  at::Tensor total_weight = OpPreparation::ApplyTensor({}, self.options(), self);
  return std::get<0>(at::nll_loss2d_forward_out(
      output, total_weight, self, target, weight, reduction, ignore_index));
}

at::Tensor & NPUNativeFunctions::multilabel_margin_loss_out(
    const at::Tensor & self,
    const at::Tensor & target,
    int64_t reduction,
    at::Tensor & output) {
  c10::SmallVector<int64_t, SIZE> outputSize;
  const auto ndims = self.dim();
  int64_t nframe;
  if (ndims <= 1) {
    nframe = 1;
  } else {
    nframe = self.size(0);
  }

  if (reduction == at::Reduction::None) {
    outputSize = {nframe};
  }
  output = OpPreparation::ApplyTensor(outputSize, self.options(), self);
  at::Tensor is_target = OpPreparation::ApplyTensor(target);
  return std::get<0>(at::multilabel_margin_loss_forward_out(output, is_target, self, target, reduction));
}

at::Tensor NPUNativeFunctions::multilabel_margin_loss(
    const at::Tensor & self,
    const at::Tensor & target,
    int64_t reduction) {
  return std::get<0>(at::multilabel_margin_loss_forward(self, target, reduction));
}

} // namespace native
} // namespace at_npu
