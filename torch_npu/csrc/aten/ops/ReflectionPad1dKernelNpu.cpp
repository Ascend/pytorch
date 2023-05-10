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

at::Tensor& NPUNativeFunctions::reflection_pad1d_out(
    const at::Tensor& self,
    at::IntArrayRef padding,
    at::Tensor& result){
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
  at::Tensor self_cp = self.unsqueeze(0);
  NPUNativeFunctions::reflection_pad2d_out(self_cp, paddings, result);
  result.squeeze_(0);
  return result;
}

at::Tensor NPUNativeFunctions::reflection_pad1d(const at::Tensor& self, at::IntArrayRef padding) {
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
  at::Tensor self_cp = self.unsqueeze(0);
  at::Tensor result = NPUNativeFunctions::reflection_pad2d(self_cp, paddings);
  result.squeeze_(0);
  return result;
}

} // namespace native
} // namespace at_npu
