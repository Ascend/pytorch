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

    at::Tensor &NPUNativeOpApiFunctions::eq_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
    {
      auto outputSize = broadcast_ops_npu_output_size(self, other);
      OpPreparation::CheckOut(
          {self, other},
          result,
          ACL_FORMAT_ND,
          result.scalar_type(),
          at::IntArrayRef(outputSize));
      EXEC_NPU_CMD(aclnnEqTensor, self, other, result);
      return result;
    }

    at::Tensor NPUNativeOpApiFunctions::eq(const at::Tensor &self, const at::Tensor &other)
    {
      at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
      at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);

      // calculate the output size
      auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize,
          formatCastOfSelf.options().dtype(at::kBool),
          ACL_FORMAT_ND);

      // calculate the output result of the NPU
      EXEC_NPU_CMD(aclnnEqTensor, formatCastOfSelf, formatCastOfOther, result);
      return result;
    }

    at::Tensor &eq_out_npu_scalar(at::Tensor &result, const at::Tensor &self, at::Scalar other)
    {
      EXEC_NPU_CMD(aclnnEqScalar, self, other, result);
      return result;
    }

    at::Tensor NPUNativeOpApiFunctions::eq(const at::Tensor &self, const at::Scalar& other)
    {
      at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);

      // calculate the output size
      auto outputSize = input_same_output_size(formatCastOfSelf);

      // construct the output tensor of the NPU
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          outputSize,
          formatCastOfSelf.options().dtype(at::kBool),
          ACL_FORMAT_ND);

      // calculate the output result of the NPU
      eq_out_npu_scalar(result, formatCastOfSelf, other);
      return result;
    }

    at::Tensor &NPUNativeOpApiFunctions::eq_out(const at::Tensor &self, const at::Scalar& other, at::Tensor &result)
    {
      OpPreparation::CheckOut(
          {self},
          result,
          ACL_FORMAT_ND,
          result.scalar_type(),
          self.sizes());
          
      eq_out_npu_scalar(result, self, other);

      return result;
    }


} // namespace native
} // namespace at_npu
