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

Tensor& glu_out_npu(Tensor& result, const Tensor& self, int64_t dim) {
  // The reason for using small operator combinations is to maintain consistency with glu_backward 
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);
  
  // According to the axis indicated by dim, it is divided into firstHalf, secondHalf
  auto chunkedInput = self.chunk(2, dim);
  Tensor firstHalf = chunkedInput[0];
  Tensor secondHalf = chunkedInput[1];

  // result = firstHalf * (sigmoid(secondHalf))
  result = firstHalf.mul(secondHalf.sigmoid());
  return result;
}

Tensor glu_npu(const Tensor& self, int64_t dim) {
  // calculate the output size
  auto outputSize = glu_npu_output_size(self, dim);
  // construct the output tensor of the NPU
  Tensor result =
      at::empty_with_format(outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
  // calculate the output result of the NPU
  glu_out_npu(result, self, dim);
  return result;
  
}
}  // namespace native
}  // namespace at