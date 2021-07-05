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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& cholesky_out_npu(
    Tensor & y, 
    const Tensor & x, 
    bool upper) {
  TORCH_CHECK(
      upper == false,
      "cholesky: The upper parameter currently only supports upper == false");

  OpCommand cmd;
  cmd.Name("Cholesky")
     .Input(x)
     .Output(y)
     .Run();
    return y;
}

Tensor cholesky_npu(const Tensor& x, bool upper) {
  Tensor formatCastOfX = x.npu_format_cast(ACL_FORMAT_NCHW);
  // calculate the output size
  auto outputSize = input_same_output_size(formatCastOfX);

  // construct the output tensor of the NPU
  Tensor y = at::empty_with_format(
      outputSize, formatCastOfX.options(), ACL_FORMAT_NCHW);

  // calculate the output result of the NPU
  cholesky_out_npu(y, formatCastOfX, upper);

  return y;
}

} // namespace native
} // namespace at