// Copyright (c) 2021 Huawei Technologies Co., Ltd
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

at::Tensor& uniform_out_npu(
    const at::Tensor& self,
    double from,
    double to,
    c10::optional<at::Generator> gen_,
    at::Tensor& result) {
  OpCommand cmd;
  cmd.Name("Uniform")
    .Input(self)
    .Output(result)
    .Attr("from", static_cast<float>(from))
    .Attr("to", static_cast<float>(to))
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::uniform_(at::Tensor& self, double from, double to, c10::optional<at::Generator> gen_) {
  // TODO(Ascend): The operator needs to use fp32 for calculation.
  at::Tensor selfCopy = self;
  if (self.scalar_type() == at::ScalarType::Half) {
    selfCopy = self.to(at::ScalarType::Float);
  }

  if (!NpuUtils::check_match(&selfCopy)) {
    at::Tensor selfContiguous = NpuUtils::format_contiguous(selfCopy);
    at::Tensor result = uniform_out_npu(selfContiguous, from, to, gen_, selfContiguous);
    NpuUtils::format_fresh_view(selfCopy, result);
  } else {
    uniform_out_npu(selfCopy, from, to, gen_, selfCopy);
  }
  self.copy_(selfCopy);
  
  return self;
}

} // namespace native
} // namespace at_npu
