// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::cumsum_out(const at::Tensor& self, int64_t dim,
                                                c10::optional<at::ScalarType> dtype, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnCumsum, NPUNativeFunctions::cumsum_out(self, dim, dtype, result));
  OpPreparation::CheckOut({self}, result, CalcuOpUtil::GetTensorNpuFormat(result),
                          result.scalar_type(), self.sizes());

  aclDataType dtypeNew = ACL_DT_UNDEFINED;
  if (!dtype.has_value()) {
    dtypeNew = CalcuOpUtil::ConvertToAclDataType(result.scalar_type());
  } else {
    dtypeNew = CalcuOpUtil::ConvertToAclDataType(dtype.value());
  }

  EXEC_NPU_CMD(aclnnCumsum, self, dim, dtypeNew, result);
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::cumsum_out(const at::Tensor& self, at::Dimname dim,
                                                c10::optional<at::ScalarType> dtype, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnCumsum, NPUNativeFunctions::cumsum_out(self, dim, dtype, result));
  return NPUNativeOpApiFunctions::cumsum_out(self, dimname_to_position(self, dim), dtype, result);
}

at::Tensor NPUNativeOpApiFunctions::cumsum(const at::Tensor& self, int64_t dim,
                                            c10::optional<at::ScalarType> dtype) {
  DO_COMPATIBILITY(aclnnCumsum, NPUNativeFunctions::cumsum(self, dim, dtype));
  at::Tensor result = OpPreparation::ApplyTensor(self);

  NPUNativeOpApiFunctions::cumsum_out(self, dim, dtype, result);
  return result;
}

} // namespace native
} // namespace at_npu
