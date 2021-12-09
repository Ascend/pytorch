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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor nll_loss_npu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  return std::get<0>(
      at::nll_loss_forward(self, target, weight, reduction, ignore_index));
}

Tensor& nll_loss_out_npu(
    Tensor& output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  Tensor total_weight = at::empty_with_format(
      {},
      self.options(),
      CalcuOpUtil::get_tensor_npu_format(self));
  return std::get<0>(at::nll_loss_forward_out(
      output, total_weight, self, target, weight, reduction, ignore_index));
}

Tensor nll_loss2d_npu(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  return std::get<0>(
      at::nll_loss2d_forward(self, target, weight, reduction, ignore_index));
}

Tensor& nll_loss2d_out_npu(
    Tensor& output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  Tensor total_weight = at::empty_with_format(
      {},
      self.options(),
      CalcuOpUtil::get_tensor_npu_format(self));
  return std::get<0>(at::nll_loss2d_forward_out(
      output, total_weight, self, target, weight, reduction, ignore_index));
}

Tensor & multilabel_margin_loss_out_npu(
    Tensor & output, 
    const Tensor & self,
    const Tensor & target, 
    int64_t reduction) {
  SmallVector<int64_t, SIZE> outputSize;
  const auto ndims = self.dim();
  int64_t nframe;
  if (ndims <= 1) {
    nframe = 1;
  } else {
    nframe = self.size(0);
  }

  if (reduction == Reduction::None) {
    outputSize = {nframe};
  }
  output = at::empty_with_format(outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  Tensor is_target = at::empty_with_format(
      target.sizes(), target.options(), CalcuOpUtil::get_tensor_npu_format(target));  
  return std::get<0>(at::multilabel_margin_loss_forward_out(output, is_target, self, target, reduction));
}

Tensor multilabel_margin_loss_npu(
    const Tensor & self, 
    const Tensor & target, 
    int64_t reduction) {
  return std::get<0>(at::multilabel_margin_loss_forward(self, target, reduction));
}

} // namespace native
} // namespace at
