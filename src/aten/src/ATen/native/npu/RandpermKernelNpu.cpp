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
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor randperm_npu(int64_t n, const TensorOptions& options) {
  return native::randperm(n, nullptr, options);
}

Tensor randperm_npu(
    int64_t n,
    Generator* generator,
    const TensorOptions& options) {
  Tensor result = at::empty_with_format({n}, options, ACL_FORMAT_NCHW);
  return at::randperm_out(result, n, generator);
}

Tensor& randperm_out_npu(Tensor& result, int64_t n) {
  return at::randperm_out(result, n, nullptr);
}

Tensor& randperm_out_npu(Tensor& result, int64_t n, Generator* generator) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);

  OpCommand cmd;
  cmd.Name("Randperm")
       .Output(result)
       .Run();
  return result;
}

} // namespace native
} // namespace at
