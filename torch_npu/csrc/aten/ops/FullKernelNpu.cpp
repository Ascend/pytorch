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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::full_out(at::IntArrayRef size, const at::Scalar& fill_value, at::Tensor& out) {
  OpPreparation::CheckOut(
      {},
      out,
      out,
      size);
  fill_(out, fill_value);
  return out;
}

at::Tensor NPUNativeFunctions::full(
    at::IntArrayRef size,
    const at::Scalar& fill_value,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  c10::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
                                          .device(device_opt)
                                          .layout(layout_opt)
                                          .pinned_memory(pin_memory_opt);
  at::Tensor result = OpPreparation::ApplyTensorWithSizes(size, option);
  return result.fill_(fill_value);
}

}
}