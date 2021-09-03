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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;
Tensor& full_out_npu(Tensor& out, IntArrayRef size, Scalar fill_value) {
  // construct the output tensor of the NPU
  at::native::fill_npu_(out, fill_value);
  return out;	
}

Tensor full_npu(
    IntArrayRef size, 
    Scalar fill_value,
    optional<DimnameList> names, 
    const TensorOptions& options) {
  Tensor result = at::empty_with_format(size, options);
  return result.fill_(fill_value);
}
}
}

