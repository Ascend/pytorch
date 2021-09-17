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

#include <torch/library.h>
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& zeros_out_npu(IntArrayRef size, Tensor& result) {
  result.resize_(size);
  return result.zero_();
}

Tensor zeros_npu(IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(size,
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt,
      ACL_FORMAT_ND);

  // calculate the output result of the NPU
  return result.zero_();
}

Tensor zeros_npu_names(
    IntArrayRef size,
    optional<DimnameList> names,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(size,
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt,
      ACL_FORMAT_ND);
  // calculate the output result of the NPU
  return result.zero_();
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("zeros.out", TORCH_FN(zeros_out_npu));
  m.impl("zeros", TORCH_FN(zeros_npu));
  m.impl("zeros.names",TORCH_FN(zeros_npu_names));
}
} // namespace native
} // namespace at