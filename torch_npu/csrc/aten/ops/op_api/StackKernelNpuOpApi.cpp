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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include <third_party/acl/inc/op_proto/split_combination_ops.h>
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>

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
  auto outputSize = stack_output_size(tensors, dim);

  OpPreparation::CheckOut(
      {tensors[0]}, 
      result, 
      ACL_FORMAT_ND, 
      tensors[0].scalar_type(), 
      outputSize); 

  EXEC_NPU_CMD(aclnnStack, tensors, dim, result);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::stack(at::TensorList tensors, int64_t dim) {
  auto outputSize = stack_output_size(tensors, dim);

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      tensors[0].options(),
      ACL_FORMAT_ND);

  EXEC_NPU_CMD(aclnnStack, tensors, dim, result);

  return result;
}

} // namespace native
} // namespace at_npu
