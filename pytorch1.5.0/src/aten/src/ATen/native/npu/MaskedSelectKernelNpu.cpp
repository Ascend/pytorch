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

Tensor& masked_select_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& mask) {
  Tensor maskBool = mask.dtype() == at::kBool ? mask : mask.to(at::kBool);
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

Tensor& masked_select_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& mask) {
  Tensor maskCast = mask.clone();
  if (maskCast.sizes() != self.sizes()) {
    maskCast = broadcast_npu(mask, self.sizes());
  }
  auto outputSize = maskCast.numel();

  OpPreparation::CheckOut(
      {self, maskCast},
      result,
      self,
      outputSize);

  masked_select_out_npu_nocheck(result, self, maskCast);
  return result;
}

Tensor masked_select_npu(
    const Tensor& self,
    const Tensor& mask) {
  Tensor maskCast = mask;
  if (maskCast.sizes() != self.sizes()) {
    maskCast = broadcast_npu(mask, self.sizes());
  }
  Tensor result = OpPreparation::ApplyTensor(self, maskCast.numel());
  masked_select_out_npu_nocheck(result, self, maskCast);

  return result;
}

} // namespace native
} // namespace at
