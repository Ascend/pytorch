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
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeFunctions::ones_out(at::IntArrayRef size, at::Tensor& result) {
  return NPUNativeFunctions::one_(result);
}

at::Tensor NPUNativeFunctions::ones(at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  // construct the output tensor of the NPU
  auto device =  device_or_default(device_opt);
  at::TensorOptions option;
  option = option.dtype(dtype_opt)
                 .layout(layout_opt)
                 .device(device)
                 .pinned_memory(pin_memory_opt);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(size, option, ACL_FORMAT_ND);

  // calculate the output result of the NPU
  return NPUNativeFunctions::one_(result);
}

at::Tensor NPUNativeFunctions::ones(
    at::IntArrayRef size,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  // construct the output tensor of the NPU
  auto device =  device_or_default(device_opt);
  at::TensorOptions option;
  option = option.dtype(dtype_opt)
                 .layout(layout_opt)
                 .device(device)
                 .pinned_memory(pin_memory_opt);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(size, option, ACL_FORMAT_ND);

  // calculate the output result of the NPU
  return NPUNativeFunctions::one_(result);
}

} // namespace native
} // namespace at_npu