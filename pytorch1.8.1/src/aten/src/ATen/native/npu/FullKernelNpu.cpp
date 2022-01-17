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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;
Tensor& full_out_npu(IntArrayRef size, Scalar fill_value, Tensor& out) {
  OpPreparation::CheckOut(
      {},
      out,
      out,
      size);
  // construct the output tensor of the NPU
  at::native::fill_(out, fill_value);
  return out;	
}

Tensor full_name_npu(
    IntArrayRef size, 
    Scalar fill_value,
    optional<DimnameList> names,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  TensorOptions option;
  option.dtype(dtype_opt);
  auto device = device_or_default(device_opt);
  option.device(device);
  option.layout(layout_opt);
  option.pinned_memory(pin_memory_opt);
  Tensor result = OpPreparation::ApplyTensorWithSizes(size, option);
  return result.fill_(fill_value);
}

Tensor full_npu(
    IntArrayRef size,
    Scalar fill_value,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  TensorOptions option;
  option.dtype(dtype_opt);
  auto device = device_or_default(device_opt);
  option.device(device);
  option.layout(layout_opt);
  option.pinned_memory(pin_memory_opt);
  Tensor result = OpPreparation::ApplyTensorWithSizes(size, option);
  return result.fill_(fill_value);
}

TORCH_LIBRARY_IMPL(aten, NPU, m){
  m.impl("full.names", TORCH_FN(full_name_npu));
  m.impl("full.out", TORCH_FN(full_out_npu));
  m.impl("full", TORCH_FN(full_npu));
}
}
}