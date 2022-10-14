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

at::Tensor celu_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar alpha) {
  OpCommand cmd;
  cmd.Name("CeluV2")
        .Input(self)
        .Output(result)
        .Attr("alpha", alpha)
        .Run();

  return result;
}

at::Tensor NPUNativeFunctions::celu(const at::Tensor& self, const at::Scalar& alpha) {
  at::Tensor result = OpPreparation::ApplyTensor(self);
  celu_out_npu_nocheck(result, self, alpha);
  return result;
}

at::Tensor& NPUNativeFunctions::celu_(at::Tensor& self, const at::Scalar& alpha) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    at::Tensor result = celu_out_npu_nocheck(contiguousSelf, contiguousSelf, alpha);
    NpuUtils::format_fresh_view(self, result);
  } else {
    celu_out_npu_nocheck(self, self, alpha);
  }
  return self;
}

} // namespace native
} // namespace at_npu
