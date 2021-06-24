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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

SmallVector<int64_t, SIZE> bmm_v2_output_size(const Tensor& mat1, const Tensor& mat2) {
  auto dim_tensor1 = mat1.dim();
  auto dim_tensor2 = mat2.dim();
  TORCH_CHECK(dim_tensor1 > 2, "mat1's dim must be greater than 2");
  TORCH_CHECK(dim_tensor2 >= 2, "mat2's dim must be greater than or equal to 2");
  if (dim_tensor2 == 2) {
    auto output_size(array_to_small_vector(mat1.sizes().slice(0, dim_tensor1-1)));
    output_size.emplace_back(mat2.size(-1));
    return output_size;
  } else {
    TORCH_CHECK(dim_tensor1 == dim_tensor2, "if mat2's dim > 2, mat1's and mat2's batch size must be same");
    IntArrayRef batch_tensor1(mat1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
    SmallVector<int64_t, SIZE> output_size = array_to_small_vector(batch_tensor1);
    output_size.emplace_back(mat1.size(-2));
    output_size.emplace_back(mat2.size(-1));
    return output_size;
  }
}


Tensor bmm_v2_npu(const Tensor& self, const Tensor& mat2) {
	auto outputSize = bmm_v2_output_size(self, mat2);
	Tensor result;

  if ((self.scalar_type() == ScalarType::Float || self.scalar_type() == ScalarType::Half)) {
    result = at::empty_with_format(outputSize, self.options(), ACL_FORMAT_FRACTAL_NZ);
  } else {
    result = at::empty_with_format(outputSize, self.options(), ACL_FORMAT_ND);
  }

  Tensor contiguousSelf = self;
  Tensor contiguousMat2 = mat2;
  if(! CalcuOpUtil::is_transpose_last_two_dims(self)){
    contiguousSelf = NpuUtils::format_contiguous(self);
  }
  if(! CalcuOpUtil::is_transpose_last_two_dims(mat2)){
    contiguousMat2 = NpuUtils::format_contiguous(mat2);
  }

  auto func1 = [&contiguousSelf]() {
      bool pass = false;
      return std::tie(pass, contiguousSelf);
  };
  auto func2 = [&contiguousMat2]() {
      bool pass = false;
      return std::tie(pass, contiguousMat2);
  };

  bool isSelfT = CalcuOpUtil::is_transpose_last_two_dims(self);
  bool isMat2T = CalcuOpUtil::is_transpose_last_two_dims(mat2);

  // executing the NPU operator
  OpCommand cmd;
  cmd.Name("BatchMatMul")
      .InputWithFunc(func1)
      .InputWithFunc(func2)
      .Output(result)
      .Attr("adj_x1", isSelfT)
      .Attr("adj_x2", isMat2T)
      .Run();

  return result;
}
} // namespace native
} // namespace at
