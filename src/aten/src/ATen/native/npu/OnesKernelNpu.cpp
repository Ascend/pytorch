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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& ones_out_npu(Tensor& result, IntArrayRef size) {
  return result.one_();
}

Tensor ones_npu(IntArrayRef size, const TensorOptions& options) {
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(size, options);

  // calculate the output result of the NPU
  return result.one_();
}

Tensor ones_npu(
    IntArrayRef size,
    optional<DimnameList> names,
    const TensorOptions& options) {
  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(size, options);

  // calculate the output result of the NPU
  return result.one_();
}

} // namespace native
} // namespace at
