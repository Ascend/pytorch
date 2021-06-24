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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/NpuUtils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor &glu_backward_out_npu(Tensor &result, const Tensor &grad_output, const Tensor &self, int64_t dim) {
  // The reason why glu_backward uses a combination of small operators is that some 
  // functions of the tbe implemented are problematic, And its accuracy compliance
  // rate is much higher than the realized tbe operator 

  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn); 

  // According to the axis indicated by dim, it is divided into firstHalf, secondHalf           
  auto chunkedInput = self.chunk(2, dim);
  Tensor firstHalf = chunkedInput[0];
  Tensor secondHalf = chunkedInput[1];

  // secondHalf = sigmoid(secondHalf)
  secondHalf = secondHalf.sigmoid();

  // grad_first = secondHalf * grad_output
  Tensor gradFirst = secondHalf.mul(grad_output);

  // grad_second = firstHalf * secondHalf * (1 - secondHalf) * grad_output
  Tensor gradSecond = firstHalf.mul(secondHalf).mul_(1-secondHalf).mul_(grad_output);

  // grad_input = gather on dim
  result = at::cat({gradFirst, gradSecond}, dim);
  return result;
}

Tensor glu_backward_npu(const Tensor &grad_output, const Tensor &self, int64_t dim) {
  auto outputSize = input_same_output_size(self);
  Tensor result = at::empty_with_format(outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  glu_backward_out_npu(result, grad_output, self, dim);
  return result;
  
}
}  // namespace native
}  // namespace at