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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {


int64_t adaptive_avg_pool3d_backward_safe_size(const at::Tensor& self){
  c10::SmallVector<int64_t, N> dims = {-3, -2, -1};
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

at::Tensor& NPUNativeFunctions::adaptive_avg_pool3d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Tensor& result){
  if (grad_output.size(grad_output.dim() - 3) == 1 && grad_output.size(grad_output.dim() - 2) == 1 &&
        grad_output.size(grad_output.dim() - 1) == 1){
    result.fill_(1.0 / adaptive_avg_pool3d_backward_safe_size(self));
    result.mul_(grad_output);
  } else {
    TORCH_CHECK(false,
        "adaptive_avg_pool3d_backward only support D=1 && H=1 && W=1 current!");
  }
  return result;
}

at::Tensor NPUNativeFunctions::_adaptive_avg_pool3d_backward(const at::Tensor& grad_output, const at::Tensor& self){
  at::Tensor result = OpPreparation::ApplyTensor(self);
  NPUNativeFunctions::adaptive_avg_pool3d_backward_out(grad_output, self, result);
  return result;
}

} // native
} // at_npu
