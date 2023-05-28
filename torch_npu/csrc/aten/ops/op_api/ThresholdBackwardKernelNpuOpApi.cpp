/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {
//- func: threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor
at::Tensor NPUNativeOpApiFunctions::threshold_backward(const at::Tensor &gradOutput, const at::Tensor &self, const at::Scalar &threshold) {
  // 根据推导出的输出shape申请输出tensor
  auto result = OpPreparation::ApplyTensorWithFormat(self.sizes(), self.options(), CalcuOpUtil::GetTensorNpuFormat(self));

  if (CalcuOpUtil::GetScalarFloatValue(threshold) != 0) {
    // use 5HD in Relu
    at::Tensor gradOutputIn = gradOutput;
    if ((torch_npu::NPUBridge::GetNpuStorageImpl(gradOutput)->npu_desc_.npu_format_ == ACL_FORMAT_NCHW) &&
         (torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.npu_format_ == ACL_FORMAT_NC1HWC0)) {
        gradOutputIn = NPUNativeFunctions::npu_format_cast(gradOutput, ACL_FORMAT_NC1HWC0);
    }
    OpCommand cmd;
    cmd.Name("ThresholdGradV2D")
        .Input(gradOutputIn)
        .Input(self)
        .Output(result)
        .Attr("threshold", threshold)
        .Run();
  } else {
    // 固定写法，调用hostapi，下发计算任务
    EXEC_NPU_CMD(aclnnThresholdBackward, gradOutput, self, threshold, result);
  }
  return result;
}
} // namespace native 
} // namespace at_npu 