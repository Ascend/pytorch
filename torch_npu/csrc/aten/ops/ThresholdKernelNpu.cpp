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
at::Tensor& NPUNativeFunctions::threshold_out(
    const at::Tensor& self,
    at::Scalar threshold,
    at::Scalar value,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("ThresholdV2D")
      .Input(self)
      .Output(result)
      .Attr("threshold", threshold)
      .Attr("value", value)
      .Run();

  return result;
}

at::Tensor NPUNativeFunctions::threshold(const at::Tensor& self, at::Scalar threshold, at::Scalar value) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  NPUNativeFunctions::threshold_out(self, threshold, value, result);
  return result;
}

at::Tensor& NPUNativeFunctions::threshold_(at::Tensor& self, at::Scalar threshold, at::Scalar value) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor selfContiguous = NpuUtils::format_contiguous(self);
    at::Tensor result =
        NPUNativeFunctions::threshold_out(selfContiguous, threshold, value, selfContiguous);
    NpuUtils::format_fresh_view(self, result);
  } else {
    NPUNativeFunctions::threshold_out(self, threshold, value, self);
  }

  return self;
}

} // namespace native
} // namespace at_npu