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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

tuple<at::Tensor&, at::Tensor&> NPUNativeOpApiFunctions::topk_out(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    at::Tensor& values,
    at::Tensor& indices) {
  at::Tensor selfCp = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = topk_npu_output_size(selfCp, k, dim, largest, sorted);

  c10::SmallVector<int64_t, SIZE> indicesSize = outputSize;
  at::Tensor indices_tmp;

  OpPipeWithMultiOut<at::Tensor&, at::Tensor&> pipe(values, indices_tmp);
  pipe.FixOutputSizeAndFormat<0>({selfCp}, selfCp, CalcuOpUtil::GetTensorNpuFormat(selfCp), outputSize)
      .ApplyOutputWithSpecailParams<1>(indicesSize, selfCp.options().dtype(at::kLong), ACL_FORMAT_ND);

  EXEC_NPU_CMD(aclnnTopk, self, k, dim, largest, sorted, values, indices_tmp);

  return pipe.ReflushOutputDtype<1>(at::ScalarType::Long)
             .FixOutputExceptDtype<1>({selfCp}, ACL_FORMAT_ND, at::ScalarType::Long, indicesSize)
             .FixOutputWithReplace<1>(indices)
             .ReturnRef<at::Tensor&, at::Tensor&>();
}

tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::topk(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  at::Tensor selfCp = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = topk_npu_output_size(selfCp, k, dim, largest, sorted);

  at::Tensor values = OpPreparation::ApplyTensorWithFormat(
      outputSize, selfCp.options(), CalcuOpUtil::GetTensorNpuFormat(selfCp));
  at::Tensor indices = OpPreparation::ApplyTensorWithFormat(
      outputSize, selfCp.options().dtype(at::kLong), CalcuOpUtil::GetTensorNpuFormat(selfCp));

  EXEC_NPU_CMD(aclnnTopk, selfCp, k, dim, largest, sorted, values, indices);
  return tuple<at::Tensor, at::Tensor>(values, indices);
}
} // namespace native
} // namespace at_npu
