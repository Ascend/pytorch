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

#include "c10/npu/npu_log.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<int64_t, SIZE> logdet_npu_output_size(const Tensor& self) {
  c10::SmallVector<int64_t, SIZE> dimVec;
  if (self.dim() > 2) {
    for (int i = 0; i < self.dim() - 2; i++) {
      dimVec.push_back(self.size(i));
    }
  }
  return dimVec;
}

Tensor& logdet_out_npu(Tensor& result, const Tensor& self) {
  OpCommand cmd;
  cmd.Name("LogDet")
     .Input(self)
     .Output(result)
     .Run();

  return result;
}

Tensor logdet_npu(const Tensor& self) {
  // calculate the output size
  auto outputSize = logdet_npu_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensorWithSizes(outputSize, self.options());
  // calculate the output result of the NPU
  logdet_out_npu(result, self);
  return result;
}
} // namespace native
} // namespace at
