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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
at::Tensor& eye_out_npu_nocheck(at::Tensor& result, int64_t n, int64_t m) {
  OpCommand cmd;
  cmd.Name("Eye")
    .Output(result)
    .Attr("num_rows", n)
    .Attr("num_columns", m)
    .Run();
    
  return result;
}

at::Tensor& NPUNativeFunctions::eye_out(int64_t n, at::Tensor& result) {
  return NPUNativeFunctions::eye_out(n, -1, result);
}

at::Tensor& NPUNativeFunctions::eye_out(int64_t n, int64_t m, at::Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);

  if (m < 0) {
    m = n;
  }

  result.resize_({n, m});
  eye_out_npu_nocheck(result, n, m);
  return result;
}

at::Tensor NPUNativeFunctions::eye(
    int64_t n,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = device_or_default(device_opt);
  at::TensorOptions option;
  option = option.dtype(dtype_opt)
                 .layout(layout_opt)
                 .device(device)
                 .pinned_memory(pin_memory_opt);

  // get the output size
  c10::SmallVector<int64_t, N> outputSize = {n, n};

  // The operator does not support the bool type and needs to be converted to an integer.
  at::Tensor result = (option.dtype() == at::kBool)
      ? OpPreparation::ApplyTensorWithFormat(outputSize, option.dtype(at::ScalarType::Int), ACL_FORMAT_ND)
      : OpPreparation::ApplyTensorWithFormat(outputSize, option, ACL_FORMAT_ND);

  NPUNativeFunctions::eye_out(n, result);
  
  if (option.dtype() == at::kBool) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
  }

  return result;
}

at::Tensor NPUNativeFunctions::eye(
    int64_t n,
    int64_t m,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  auto device = device_or_default(device_opt);
  at::TensorOptions option;
  option = option.dtype(dtype_opt)
                 .layout(layout_opt)
                 .device(device)
                 .pinned_memory(pin_memory_opt);

  // get the output size
  c10::SmallVector<int64_t, N> outputSize = {n, m};

  // The operator does not support the bool type and needs to be converted to an integer.
  at::Tensor result = (option.dtype() == at::kBool)
      ? OpPreparation::ApplyTensorWithFormat(outputSize, option.dtype(at::ScalarType::Int), ACL_FORMAT_ND)
      : OpPreparation::ApplyTensorWithFormat(outputSize, option, ACL_FORMAT_ND);

  eye_out_npu_nocheck(result, n, m);
  
  if (option.dtype() == at::kBool) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
  }

  return result;
}
} // namespace native
} // namespace at_npu