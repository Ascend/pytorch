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

#include <ATen/ExpandUtils.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::SmallVector<int64_t, SIZE> masked_select_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& mask) {
  c10::MaybeOwned<at::Tensor> maskCast;
  c10::MaybeOwned<at::Tensor> selfCast;
  std::tie(maskCast, selfCast) = expand_outplace(mask, self);
  auto outputSize = {maskCast->sum().item().toLong()};
  return outputSize;
}

at::Tensor& masked_select_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mask) {
  at::Tensor maskBool = mask.dtype() == at::kBool ? mask : NPUNativeFunctions::npu_dtype_cast(mask, at::kBool);
  c10::SmallVector<int64_t, N> output_sync_idx = {0};
  OpCommand cmd;
  cmd.Sync(output_sync_idx)
      .Name("MaskedSelect")
      .Input(self)
      .Input(maskBool)
      .Output(result)
      .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::masked_select_out(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Tensor& result) {
  at::Tensor maskCast = mask.clone();
  auto outputSize = masked_select_npu_output_size(self, maskCast);
  OpPreparation::CheckOut(
      {self, maskCast},
      result,
      self,
      outputSize);

  masked_select_out_npu_nocheck(result, self, maskCast);
  return result;
}

at::Tensor NPUNativeFunctions::masked_select(
    const at::Tensor& self,
    const at::Tensor& mask) {
  auto outputSize = masked_select_npu_output_size(self, mask);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  masked_select_out_npu_nocheck(result, self, mask);
  return result;
}

} // namespace native
} // namespace at_npu
