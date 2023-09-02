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

#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor & NPUNativeOpApiFunctions::multilabel_margin_loss_out(
    const at::Tensor & self,
    const at::Tensor & target,
    int64_t reduction,
    at::Tensor & output) {
  DO_COMPATIBILITY(aclnnMultilabelMarginLoss,
                   NPUNativeFunctions::multilabel_margin_loss_out(self, target, reduction,
                   output));
  c10::SmallVector<int64_t, SIZE> is_target_size;
  if (self.dim() == 2 && self.size(0) ==0) {
    is_target_size = {self.size(0)};
  } else {
    is_target_size = array_to_small_vector(target.sizes());
  }
  auto is_target = OpPreparation::ApplyTensorWithoutFormat(self, is_target_size);
  return std::get<0>(NPUNativeOpApiFunctions::multilabel_margin_loss_forward_out(self, target, reduction,output, is_target));
}

at::Tensor NPUNativeOpApiFunctions::multilabel_margin_loss(
    const at::Tensor & self,
    const at::Tensor & target,
    int64_t reduction) {
  return std::get<0>(at::multilabel_margin_loss_forward(self, target, reduction));
}

std::tuple<at::Tensor&, at::Tensor&> NPUNativeOpApiFunctions::multilabel_margin_loss_forward_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& output,
    at::Tensor& is_target) {
  DO_COMPATIBILITY(aclnnMultilabelMarginLoss,
                   NPUNativeFunctions::multilabel_margin_loss_forward_out(self, target, reduction,
                   output, is_target));
  c10::SmallVector<int64_t, SIZE> output_size;
  if (self.dim() == 2 && reduction == at::Reduction::None) {
    output_size = {self.size(0)};
  }
  c10::SmallVector<int64_t, SIZE> is_target_size;
  if (self.dim() == 2 && self.size(0) ==0) {
    is_target_size = {self.size(0)};
  } else {
    is_target_size = array_to_small_vector(target.sizes());
  }
  OpPreparation::CheckOut({self, target}, output, output, output_size);
  OpPreparation::CheckOut({self, target}, is_target, is_target, is_target_size);

  EXEC_NPU_CMD(aclnnMultilabelMarginLoss, self, target, reduction, output, is_target);
  return std::tuple<at::Tensor&, at::Tensor&>(output, is_target);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::multilabel_margin_loss_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  DO_COMPATIBILITY(aclnnMultilabelMarginLoss,
                   NPUNativeFunctions::multilabel_margin_loss_forward(self, target, reduction));
  c10::SmallVector<int64_t, SIZE> output_size;
  if (self.dim() == 2 && reduction == at::Reduction::None) {
    output_size = {self.size(0)};
  }
  c10::SmallVector<int64_t, SIZE> is_target_size;
  if (self.dim() == 2 && self.size(0) ==0) {
    is_target_size = {self.size(0)};
  } else {
    is_target_size = array_to_small_vector(target.sizes());
  }
  auto output = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  auto is_target = OpPreparation::ApplyTensorWithoutFormat(self, is_target_size);

  NPUNativeOpApiFunctions::multilabel_margin_loss_forward_out(
      self, target, reduction, output, is_target);
  return std::make_tuple(output, is_target);
}

} // namespace native
} // namespace at_npu

