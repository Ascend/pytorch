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

Tensor& softplus_out_npu(
    const Tensor& self,
    Scalar beta,
    Scalar threshold,
    Tensor& result) {
  OpPreparation::CheckMemory({self}, {result});
  OpCommand cmd;
  cmd.Name("SoftplusV2")
      .Input(self)
      .Output(result)
      .Attr("beta", beta)
      .Attr("threshold", threshold)
      .Run();
  return result;
}

Tensor softplus_npu(
    const Tensor& self,
    Scalar beta,
    Scalar threshold) {
  Tensor result = OpPreparation::ApplyTensor(self);

  softplus_out_npu(self, beta, threshold, result);
  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("softplus", TORCH_FN(softplus_npu));
  m.impl("softplus.out", TORCH_FN(softplus_out_npu));
}
} // namespace native
} // namespace at
