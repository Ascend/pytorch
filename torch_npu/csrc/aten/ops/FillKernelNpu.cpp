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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& fill_out_npu(at::Tensor& result, at::Tensor& self, const at::Tensor& other) {
  OpCommand cmd;
  cmd.Name("Fill");
  if (self.dim() == 0) {
    c10::SmallVector<int64_t, N> dims = {1};
    cmd.Input(dims, at::kLong);
  } else {
    cmd.Input(self.sizes(), at::kLong);
  }
  cmd.Input(other)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& fill_out_npu(at::Tensor& result, at::Tensor& self, at::Scalar value) {
  OpCommand cmd;
  cmd.Name("Fill");
  if (self.dim() == 0) {
    c10::SmallVector<int64_t, N> dims = {1};
    cmd.Input(dims, at::kLong);
  } else {
    cmd.Input(self.sizes(), at::kLong);
  }
  cmd.Input(value, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Output(result)
      .Run();
  return result;
}

at::Tensor& NPUNativeFunctions::fill_(at::Tensor& self, const at::Tensor& other) {
  auto other_dim = other.dim();
  TORCH_CHECK(other_dim <= 1, "fill_ only supports 0 or 1 dimension value tensor but got tensor with ",
      other_dim, " dimension.");
  if (other_dim == 0 && !at_npu::key::isDeviceTensor(other)) {
    fill_out_npu(self, self, other.item());
  } else {
    fill_out_npu(self, self, other);
  }
  return self;
}

at::Tensor& NPUNativeFunctions::fill_(at::Tensor& self, const at::Scalar& value) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = fill_out_npu(contiguousSelf, contiguousSelf, value);
    NpuUtils::format_fresh_view(self, result);
  } else {
    fill_out_npu(self, self, value);
  }

  return self;
}

} // namespace native
} // namespace at_npu
