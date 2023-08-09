// Copyright (c) 2023 Huawei Technologies Co., Ltds
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

#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"

namespace at_npu {
namespace native {

c10::SmallVector<at::Tensor, N> complex_compute_pre_check_split(const at::Tensor& input) {
  c10::SmallVector<at::Tensor, N> input_split;
  if (input.is_complex()) {
    auto input_r = complex_compute(input);
    input_split = at::chunk(input_r, 2, -1);
    input_split[0].squeeze_(-1);
    input_split[1].squeeze_(-1);
  } else {
    input_split.emplace_back(input);
    input_split.emplace_back(at::zeros(input.sizes(), input.options()));
  }
  return input_split;
}

c10::SmallVector<at::Tensor, N> complex_compute_split(const at::Tensor& input) {
  c10::SmallVector<at::Tensor, N> input_split;
  auto input_r = complex_compute(input);
  input_split = at::chunk(input_r, 2, -1);
  return input_split;
}

at::Tensor complex_compute(const at::Tensor& input) {
  c10::SmallVector<at::Tensor, N> input_split;
  auto input_r = at::native::view_as_real(input).clone();
  torch_npu::NPUStorageDesc &input_r_desc = torch_npu::NPUBridge::GetNpuStorageImpl(input_r)->npu_desc_;
  input_r_desc.base_sizes_ = input_r.sizes();
  input_r_desc.base_strides_ = input_r.strides();
  return input_r;
}

} // namespace native
} // namespace at_npu
