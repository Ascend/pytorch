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

Tensor celu_out_npu_nocheck(Tensor& result, const Tensor& self, Scalar alpha) {
  float alpha3 = 1.0;

  OpCommand cmd;
  cmd.Name("Celu")
        .Input(self)
        .Output(result)
        .Attr("alpha1", alpha)
        .Attr("alpha2", alpha)
        .Attr("alpha3", alpha3)
        .Run();

  return result;
}

Tensor celu_npu(const Tensor& self, Scalar alpha) {
  Tensor result = OpPreparation::ApplyTensor(self);

  celu_out_npu_nocheck(result, self, alpha);

  return result;
}

Tensor& celu_npu_(Tensor& self, Scalar alpha) {
  OpPreparation::CheckMemory({self}, {self});

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = celu_out_npu_nocheck(contiguousSelf, contiguousSelf, alpha);
    NpuUtils::format_fresh_view(self, result);
  } else {
    celu_out_npu_nocheck(self, self, alpha);
  }

  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("celu", TORCH_FN(celu_npu));
  m.impl("celu_", TORCH_FN(celu_npu_));
}
} // namespace native
} // namespace at
