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

Tensor& threshold_out_npu(
    Tensor& result,
    const Tensor& self,
    Scalar threshold,
    Scalar value) {
  OpCommand cmd;
  cmd.Name("ThresholdV2")
      .Input(self)
      .Input(threshold, self.scalar_type())
      .Input(value, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

Tensor threshold_npu(const Tensor& self, Scalar threshold, Scalar value) {
  Tensor result = OpPreparation::ApplyTensor(self);
  threshold_out_npu(result, self, threshold, value);
  return result;
}

Tensor& threshold_npu_(Tensor& self, Scalar threshold, Scalar value) {
  if (!NpuUtils::check_match(&self)) {
    Tensor selfContiguous = NpuUtils::format_contiguous(self);
    Tensor result =
        threshold_out_npu(selfContiguous, selfContiguous, threshold, value);
    NpuUtils::format_fresh_view(self, result);
  } else {
    threshold_out_npu(self, self, threshold, value);
  }

  return self;
}

} // namespace native
} // namespace at