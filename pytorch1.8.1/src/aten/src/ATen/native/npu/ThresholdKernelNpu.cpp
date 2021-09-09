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

#include <torch/script.h>
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& threshold_out_npu(
    const Tensor& self,
    Scalar threshold,
    Scalar value,
    Tensor& result) {
  OpCommand cmd;
  cmd.Name("ThresholdV2D")
      .Input(self)
      .Output(result)
      .Attr("threshold", threshold)
      .Attr("value", value)
      .Run();

  return result;
}

Tensor threshold_npu(const Tensor& self, Scalar threshold, Scalar value) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  threshold_out_npu(self, threshold, value, result);
  return result;
}

Tensor& threshold_npu_(Tensor& self, Scalar threshold, Scalar value) {
  if (!NpuUtils::check_match(&self)) {
    Tensor selfContiguous = NpuUtils::format_contiguous(self);
    Tensor result =
        threshold_out_npu(selfContiguous, threshold, value, selfContiguous);
    NpuUtils::format_fresh_view(self, result);
  } else {
    threshold_out_npu(self, threshold, value, self);
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("threshold", TORCH_FN(threshold_npu));
  m.impl("threshold_", TORCH_FN(threshold_npu_));
  m.impl("threshold.out", TORCH_FN(threshold_out_npu));
}

} // namespace native
} // namespace at