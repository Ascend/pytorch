// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include <ATen/NamedTensorUtils.h>

#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

// c = a @ b, then
// a_grad = c_grad @ b^H
// b_grad = a^H @ c_grad
std::tuple<at::Tensor, at::Tensor> npu_matmul_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& other,
    std::array<bool,2> mask,
    at::Tensor& grad_self,
    at::Tensor& grad_other) {
  // select feasible path by checking tensor dimension
  auto dim_self = self.dim();
  auto dim_other = other.dim();

  auto size_grad = grad.sizes();
  auto size_self = self.sizes();
  auto size_other = other.sizes();

  if (dim_self == 1 && dim_other == 1) {
    grad_self = mask[0] ? other.mul(grad) : grad_self;
    grad_other = mask[1] ? self.mul(grad) : grad_other;
  } else if (dim_self == 2 && dim_other == 1) {
    grad_self = mask[0] ? grad.unsqueeze(1).mm(other.unsqueeze(0)) : grad_self;
    grad_other = mask[1] ? self.transpose(-1, -2).mm(grad.unsqueeze(1)).squeeze_(1) : grad_other;
  } else if (dim_self == 1 && dim_other == 2) {
    grad_self = mask[0] ? grad.unsqueeze(0).mm(other.transpose(-1, -2)).squeeze_(0) : grad_self;
    grad_other = mask[1] ? self.unsqueeze(1).mm(grad.unsqueeze(0)) : grad_other;
  } else if (dim_self >= 3 && (dim_other == 1 || dim_other == 2)) {
    // create a 2D-matrix from grad
    const int64_t view_size = dim_other == 1 ? 1 : size_grad[size_grad.size() - 1];
    auto unfolded_grad = (dim_other == 1 ? grad.unsqueeze(-1) : grad).contiguous().view({-1, view_size});
    if (mask[0]) {
      grad_self = unfolded_grad.mm(dim_other == 1 ? other.unsqueeze(0) : other.transpose(-1, -2)).view(size_self);
    }
    if (mask[1]) {
      // create a 2D-matrix from self
      auto unfolded_self = self.contiguous().view({-1, size_self[dim_self - 1]});
      grad_other = unfolded_self.transpose(-1, -2).mm(unfolded_grad).view(size_other);
    }
  } else if ((dim_self == 1 || dim_self == 2) && dim_other >= 3) {
    // create a 2D-matrix from grad
    const int64_t view_size = dim_self == 1 ? 1 : size_grad[size_grad.size() - 2];
    auto unfolded_grad_T =
      dim_self == 1 ? grad.view({-1, view_size}) : grad.transpose(-1, -2).contiguous().view({-1, view_size});
    if (mask[0]) {
      // create a 2D-matrix from other
      auto unfolded_other_T =
        other.transpose(-1, -2).contiguous().view({-1, size_other[dim_other - 2]}).transpose(-1, -2);
      grad_self = unfolded_other_T.mm(unfolded_grad_T).transpose(-1, -2).view(size_self);
    }
    if (mask[1]) {
      std::vector<int64_t> size_other_T(size_other.begin(), size_other.end() - 2);
      size_other_T.insert(size_other_T.end(), {size_other[dim_other - 1], size_other[dim_other - 2]});
      grad_other = unfolded_grad_T.mm(dim_self == 1 ? self.unsqueeze(0) : self).view(size_other_T).transpose(-1, -2);
    }
  } else {
    grad_self = mask[0] ? NPUNativeFunctions::matmul(grad, other.transpose(-1, -2)) : grad_self;
    grad_other = mask[1] ? NPUNativeFunctions::matmul(self.transpose(-1, -2), grad): grad_other;
  }

  return std::make_tuple(grad_self, grad_other);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::matmul_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& other,
    std::array<bool,2> mask) {
  if (!grad.defined()) {
    return std::make_tuple(at::Tensor(), at::Tensor());
  }

  at::Tensor grad_self, grad_other;

  if (!mask[0] && !mask[1]) {
    return std::make_tuple(grad_self, grad_other);
  }

  if (at_npu::key::isDeviceTensor(self) &&
      at_npu::key::isDeviceTensor(other) &&
      self.scalar_type() == at::kHalf &&
      other.scalar_type() == at::kHalf &&
      at_npu::native::env::CheckBmmV2Enable()) {
    grad_self = mask[0] ? matmul_by_bmmV2(grad, other.transpose(-1, -2)) : grad_self;
    grad_other = mask[1] ? matmul_by_bmmV2(self.transpose(-1, -2), grad) : grad_other;
    return std::make_tuple(grad_self, grad_other);
  }

  npu_matmul_backward(grad, self, other, mask, grad_self, grad_other);
  return std::make_tuple(grad_self, grad_other);
}

} // namespace native
} // namespace at_npu
