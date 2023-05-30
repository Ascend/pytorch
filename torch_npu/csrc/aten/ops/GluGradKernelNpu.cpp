// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.

// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// https://opensource.org/licenses/BSD-3-Clause

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

at::Tensor &glu_backward_npu_nocheck(const at::Tensor &grad_output,
                                     const at::Tensor &self, at::Tensor &result,
                                     int64_t dim) {
  OpCommand cmd;
  cmd.Name("GLUGrad")
      .Input(grad_output)
      .Input(self)
      .Output(result)
      .Attr("dim", dim)
      .Run();
  return result;
}

at::Tensor &NPUNativeFunctions::glu_backward_out(const at::Tensor &grad_output,
                                                 const at::Tensor &self,
                                                 int64_t dim,
                                                 at::Tensor &result) {
  auto outputSize = input_same_output_size(self);
  OpPreparation::CheckOut({grad_output, self}, 
                          result, 
                          grad_output, 
                          outputSize);

  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional Tensors");
  auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  glu_backward_npu_nocheck(grad_output, self, result, dim);
  return result;
}

at::Tensor NPUNativeFunctions::glu_backward(const at::Tensor &grad_output,
                                            const at::Tensor &self,
                                            int64_t dim) {
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional Tensors");
  auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  auto outputSize = input_same_output_size(self);
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  glu_backward_npu_nocheck(grad_output, self, result, dim);
  return result;
}
} // namespace native
} // namespace at_npu
