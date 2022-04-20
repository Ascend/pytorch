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
  OpCommand cmd;
  cmd.Name("CeluV2")
        .Input(self)
        .Output(result)
        .Attr("alpha", alpha)
        .Run();
  return result;
}

Tensor celu_out_npu(Tensor& result, const Tensor& self, Scalar alpha) {
  OpPipeWithDefinedOut pipe;
  return pipe.CheckMemory({self}, {result})
   .Func([&self, &alpha](Tensor& result){celu_out_npu_nocheck(result, self, alpha);})
   .Call(result);
}

Tensor celu_npu(const Tensor& self, Scalar alpha) {
  // construct the output tensor of the NPU
  Tensor result = OpPreparation::ApplyTensor(self);

  // calculate the output result of the NPU
  celu_out_npu(result, self, alpha);

  return result;
}

Tensor& celu_npu_(Tensor& self, Scalar alpha) {
  celu_out_npu(self, self, alpha);
  return self;
}

} // namespace native
} // namespace at
