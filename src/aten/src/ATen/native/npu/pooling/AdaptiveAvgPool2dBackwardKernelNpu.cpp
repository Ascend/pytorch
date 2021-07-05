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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

int64_t adaptive_avg_pool2d_backward_safe_size(const Tensor& self) {
  SmallVector<int64_t, N> dims = {-2, -1};

  int64_t size = 1;
  if (self.sizes().empty()) {
    return size;
  }

  for (int64_t ndim : dims) {
    ndim = CalcuOpUtil::make_wrap_dim(ndim, self.sizes().size());
    size *= self.sizes()[ndim];
  }

  return size;
}

Tensor& adaptive_avg_pool2d_backward_out_npu(
    Tensor& result,
    const Tensor& grad_output,
    const Tensor& self) {
  if (grad_output.size(grad_output.dim() - 2) == 1 && grad_output.size(grad_output.dim() - 1) == 1) {
      result.fill_(1.0 / adaptive_avg_pool2d_backward_safe_size(self));
      result.mul_(grad_output);
  } else {
  OpCommand cmd;
  cmd.Name("AdaptiveAvgPool2dGrad")
       .Input(grad_output)
       .Output(result)
       .Attr("orig_input_shape", self.sizes())
       .Run();
  }
  return result;
}

Tensor adaptive_avg_pool2d_backward_npu(
    const Tensor& grad_output,
    const Tensor& self) {
  Tensor result = OpPreparation::ApplyTensor(self);
  adaptive_avg_pool2d_backward_out_npu(result, grad_output, self);
  return result;
}

} // namespace native
} // namespace at
