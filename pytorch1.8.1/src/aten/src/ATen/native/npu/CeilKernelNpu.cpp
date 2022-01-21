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

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& ceil_out_npu_nocheck(Tensor& result, const Tensor& self) {
  OpCommand cmd;
  cmd.Name("Ceil")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

Tensor& ceil_out_npu(const Tensor& self, Tensor& result) {
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  return ceil_out_npu_nocheck(result, self);
}

Tensor ceil_npu(const Tensor& self) {
  Tensor result = OpPreparation::ApplyTensor(self);
  ceil_out_npu_nocheck(result, self);
  return result;
}

Tensor& ceil_npu_(Tensor& self) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = ceil_out_npu_nocheck(contiguousSelf, contiguousSelf);
    NpuUtils::format_fresh_view(self, result);
  } else {
    ceil_out_npu_nocheck(self, self);
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("ceil", TORCH_FN(ceil_npu));
  m.impl("ceil_", TORCH_FN(ceil_npu_));
  m.impl("ceil.out", TORCH_FN(ceil_out_npu));
}
} // namespace native
} // namespace at