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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::index_select_out(const at::Tensor& self, 
                                                      int64_t dim, 
                                                      const at::Tensor& index,
                                                      at::Tensor& result) {
  at::Tensor indexTmp(index);
  if (indexTmp.ndimension() == 0) {
    indexTmp = index.unsqueeze(0);
  }
  auto outputSize = index_select_npu_output_size(self, dim, indexTmp);
  int64_t npu_format = CalcuOpUtil::GetTensorNpuFormat(self);
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_ND;
  }
  at::Tensor input = self;
  if (self.dtype() == at::kBool) {
    input = NPUNativeFunctions::npu_dtype_cast(input, at::kInt);
  }
  OpPreparation::CheckOut(
      {input},
      result,
      npu_format,
      input.scalar_type(),
      outputSize);
  OpPipeWithDefinedOut pipe;
  result = pipe.CheckMemory({input, indexTmp}, {result})
      .Func([&input, &dim, &indexTmp](at::Tensor& result)
      {EXEC_NPU_CMD(aclnnIndexSelect, input, dim, indexTmp, result);})
      .Call(result);
  if (self.dtype() == at::kBool) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
  }
  return result;
}



at::Tensor NPUNativeOpApiFunctions::index_select(const at::Tensor& self,
                                                 int64_t dim, 
                                                 const at::Tensor& index) {
  at::Tensor indexTmp(index);
  if (indexTmp.ndimension() == 0) {
    indexTmp = index.unsqueeze(0);
  }
  auto outputSize = index_select_npu_output_size(self, dim, indexTmp);
  int64_t npu_format = CalcuOpUtil::GetTensorNpuFormat(self);
  if (outputSize.empty()) {
    npu_format = ACL_FORMAT_ND;
  }
  at::Tensor input = self;
  if (self.dtype() == at::kBool) {
    input = NPUNativeFunctions::npu_dtype_cast(input, at::kInt);
  }
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(input, outputSize, npu_format);
  EXEC_NPU_CMD(aclnnIndexSelect, input, dim, indexTmp, result);
  if (self.dtype() == at::kBool) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::kBool);
  }
  return result;
}

} // namespace native
} // namespace at_npu
