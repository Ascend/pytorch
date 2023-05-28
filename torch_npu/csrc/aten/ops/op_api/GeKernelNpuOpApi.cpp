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

    at::Tensor& NPUNativeOpApiFunctions::ge_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
      at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
      auto outputSize = formatCastOfSelf.sizes(); 
      OpPreparation::CheckOut(
          {self},
          result,
          ACL_FORMAT_ND,
          result.scalar_type(),
          outputSize);

      EXEC_NPU_CMD(aclnnGeScalar, formatCastOfSelf, other, result);
      return result;
    }

    at::Tensor NPUNativeOpApiFunctions::ge(const at::Tensor& self, const at::Scalar& other) {
      at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
      at::Tensor result = OpPreparation::ApplyTensorWithFormat(
          formatCastOfSelf.sizes(),
          formatCastOfSelf.options().dtype(at::kBool),
          ACL_FORMAT_ND);
      EXEC_NPU_CMD(aclnnGeScalar, formatCastOfSelf, other, result);
      return result;
    }

  } // namespace native
} // namespace at_npu