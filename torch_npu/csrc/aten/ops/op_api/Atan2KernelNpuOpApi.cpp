// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu
{
  namespace native
  {
    at::Tensor &NPUNativeOpApiFunctions::atan2_out(
        const at::Tensor &self,
        const at::Tensor &other,
        at::Tensor &result)
    {
      DO_COMPATIBILITY(aclnnAtan2, NPUNativeFunctions::atan2_out(self, other, result));
      auto outputSize = broadcast_ops_npu_output_size(self, other);
      OpPreparation::CheckOut(
          {self, other},
          result,
          result.scalar_type(),
          outputSize);
      EXEC_NPU_CMD(aclnnAtan2, self, other, result);
      return result;
    }

    at::Tensor NPUNativeOpApiFunctions::atan2(const at::Tensor &self, const at::Tensor &other)
    {
      DO_COMPATIBILITY(aclnnAtan2, NPUNativeFunctions::atan2(self, other));
      auto outputSize = broadcast_ops_npu_output_size(self, other);
      c10::ScalarType infer_dtype = at::native::result_type(self, other);
      at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(infer_dtype));
      EXEC_NPU_CMD(aclnnAtan2, self, other, result);
      return result;
    }

    at::Tensor &NPUNativeOpApiFunctions::atan2_(at::Tensor &self, const at::Tensor &other)
    {
      DO_COMPATIBILITY(aclnnInplaceAtan2, NPUNativeFunctions::atan2_(self, other));
      EXEC_NPU_CMD(aclnnInplaceAtan2, self, other);
      return self;
    }

  } // namespace native
} // namespace at_npu
