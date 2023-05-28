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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::all_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& result) {
  c10::SmallVector<int64_t, N> dimList = {dim};
  
  // check result for return
  auto outputSize = reduce_ops_npu_output_size(self, dimList, keepdim);
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::GetTensorNpuFormat(self),
      self.scalar_type(),
      outputSize);
  if (self.numel() == 0) {
      result.fill_(true);
      return result;
  }
  // calculate the output result of the NPU    
  at::IntArrayRef dims(dim);
  EXEC_NPU_CMD(aclnnAll, self, dims, keepdim, result);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::all(const at::Tensor& self, int64_t dim, bool keepdim) {
  if (self.dim() == 0) {
    TORCH_CHECK(dim != 0 || dim != -1,
        "The value of dim must be greater than or equal to -self.dim() and less than self.dim()");
  } else {
    TORCH_CHECK(dim >= -(self.dim()) && dim < self.dim(),
        "The value of dim must be greater than or equal to -self.dim() and less than self.dim()");
  }

  // calculate the output size
  at::IntArrayRef dims(dim);
  auto outputSize = reduce_ops_npu_output_size(self, dims, keepdim);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);
  if (self.numel() == 0) {
    result.fill_(true);
    return result;
  }
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnAll, self, dims, keepdim, result);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::all(const at::Tensor& self) {
    // calculate the output size
  at::IntArrayRef dims;
  auto outputSize = reduce_ops_npu_output_size(self, dims, false);
  
  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(self, outputSize);

if (self.numel() == 0) {
  result.fill_(true);
  return result;
}
  at::IntArrayRef dimList(CalcuOpUtil::GetDimlistForTensor(self));
  bool keepdim = false;
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnAll, self, dimList, keepdim, result);

  return result;
}

} // namespace native
} // namespace at_npu