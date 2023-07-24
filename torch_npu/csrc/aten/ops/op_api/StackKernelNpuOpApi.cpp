// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <ATen/native/TypeProperties.h>
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::SmallVector<int64_t, SIZE> stack_output_size(
    at::TensorList tensors,
    int64_t dim) {
  dim = CalcuOpUtil::MakeWrapDim(dim, tensors[0].dim() + 1);
  at::SmallVector<int64_t, SIZE> shape;
  for (int i = 0; i < dim; i++) {
    shape.emplace_back(tensors[0].size(i));
  }
  shape.emplace_back(tensors.size());
  for (int i = dim; i < tensors[0].dim(); i++) {
    shape.emplace_back(tensors[0].size(i));
  }

  return shape;
}

at::Tensor& NPUNativeOpApiFunctions::stack_out(at::TensorList tensors, int64_t dim, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnStack, NPUNativeFunctions::stack_out(tensors, dim, result));
  auto output_size = stack_output_size(tensors, dim);
  OpPreparation::CheckOut({tensors[0]}, result, tensors[0], output_size); 

  EXEC_NPU_CMD(aclnnStack, tensors, dim, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::stack(at::TensorList tensors, int64_t dim) {
  DO_COMPATIBILITY(aclnnStack, NPUNativeFunctions::stack(tensors, dim));
  auto output_size = stack_output_size(tensors, dim);
  at::ScalarType result_type = at::native::result_type(tensors);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(
      output_size,
      tensors[0].options().dtype(result_type));

  EXEC_NPU_CMD(aclnnStack, tensors, dim, result);

  return result;
}

} // namespace native
} // namespace at_npu
