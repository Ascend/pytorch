// Copyright (c) 2020 Huawei Technologies Co., Ltd
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
#include <iostream>
#include "c10/npu/npu_log.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& det_out_npu(Tensor& result, const Tensor& self) {
  auto outputSize = det_npu_output_size(self);
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      outputSize);

  OpCommand cmd;
  cmd.Name("Det")
    .Input(self)
    .Output(result) 
    .Run();
  return result;
}

Tensor det_npu(const Tensor& self) {
  auto outputSize = det_npu_output_size(self);
  Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  det_out_npu(result, self);
  return result;
}
} // namespace native
} // namespace at
