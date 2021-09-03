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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include <torch/script.h>

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& ones_out_npu(IntArrayRef size, Tensor& result) {
  return result.one_();
}

Tensor ones_npu(IntArrayRef size,     
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, ACL_FORMAT_ND);

  // calculate the output result of the NPU
  return result.one_();
}

Tensor ones_dimlist_npu(
    IntArrayRef size,
    optional<DimnameList> names,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, ACL_FORMAT_ND);
 
  // calculate the output result of the NPU
  return result.one_();
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("ones.names", TORCH_FN(ones_dimlist_npu));
  m.impl("ones", TORCH_FN(ones_npu));
  m.impl("ones.out", TORCH_FN(ones_out_npu));
}

Tensor ones(IntArrayRef size,     
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {

  // calculate the output result of the NPU
  return ones_npu(size, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

} // namespace native
} // namespace at
