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

at::Tensor NPUNativeFunctions::argmax(const at::Tensor& self, at::optional<int64_t> dim, bool keepdim) {
  at::Tensor input = dim.has_value() ? self : self.reshape({-1});
  int64_t realDim = dim.has_value() ? dim.value() : 0;
  bool realKeepDim = dim.has_value() ? keepdim : false;

  // calculate the output size
  auto outputSize = reduce_ops_npu_output_size(input, realDim, realKeepDim);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithSizes(outputSize, self.options().dtype(at::kInt));
  
  at::Scalar DimVec = realDim;
  // calculate the output result of the NPU
  OpCommand cmd;
  cmd.Name("ArgMaxV2")
      .Input(input)
      .Input(DimVec, at::ScalarType::Int, CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(result)
      .Attr("keep_dims", realKeepDim)
      .Run();

  result = result.to(at::ScalarType::Long);
  return result;
}

} // namespace native
} // namespace at_npu
