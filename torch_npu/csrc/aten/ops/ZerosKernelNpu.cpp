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

at::Tensor& NPUNativeFunctions::zeros_out(at::IntArrayRef size, at::Tensor& result) {
  result.resize_(size);
  return result.zero_();
}

at::Tensor NPUNativeFunctions::zeros(at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  at::TensorOptions option = option.dtype(dtype_opt)
                                  .layout(layout_opt)
                                  .device(device_opt)
                                  .pinned_memory(pin_memory_opt);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(size, option, ACL_FORMAT_ND);
  return result.zero_();
}

at::Tensor NPUNativeFunctions::zeros(
    at::IntArrayRef size,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  return NPUNativeFunctions::zeros(size, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

} // namespace native
} // namespace at_npu