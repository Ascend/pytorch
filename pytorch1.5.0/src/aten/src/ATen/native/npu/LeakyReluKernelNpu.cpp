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

Tensor& leaky_relu_out_npu(Tensor& result, const Tensor& self, Scalar negval) {
  OpCommand cmd;
  cmd.Name("LeakyRelu")
      .Input(self)
      .Output(result)
      .Attr("negative_slope", negval)
      .Run();

  return result;
}

Tensor leaky_relu_npu(const Tensor& self, Scalar negval) {
  Tensor result = OpPreparation::ApplyTensor(self);
  // calculate the output result of the NPU
  leaky_relu_out_npu(result, self, negval);
  return result;
}

Tensor& leaky_relu_npu_(Tensor& self, Scalar neg_val) {
  OpPreparation::CheckMemory({self}, {self});
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = leaky_relu_out_npu(contiguousSelf, contiguousSelf, neg_val);
    NpuUtils::format_fresh_view(self, result);
  } else {
    leaky_relu_out_npu(self, self, neg_val);
  }

  return self;
}

} // namespace native
} // namespace at