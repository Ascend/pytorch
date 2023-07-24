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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::upsample_nearest1d_out(const at::Tensor& self,
                                                            at::IntArrayRef output_size,
                                                            c10::optional<double> scales,
                                                            at::Tensor& result) {
  DO_COMPATIBILITY(aclnnUpsampleNearest1d, NPUNativeFunctions::upsample_nearest1d_out(self, output_size,
                                                                                      scales, result));
  c10::SmallVector<int64_t, SIZE> out_size = upsample_linear1d_npu_output_size(self, output_size, false, scales);
  OpPreparation::CheckOut({self}, result, self, out_size);

  EXEC_NPU_CMD(aclnnUpsampleNearest1d, self, output_size, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::upsample_nearest1d(const at::Tensor& input,
                                                       c10::optional<at::IntArrayRef> output_size,
                                                       c10::optional<at::ArrayRef<double>> scale_factors) {
  DO_COMPATIBILITY(aclnnUpsampleNearest1d, NPUNativeFunctions::upsample_nearest1d(input, output_size, scale_factors));
  auto compute_size = CalcuOpUtil::ComputeOutputSize(input.sizes(), output_size, scale_factors);
  auto scales_w = CalcuOpUtil::GetScaleValue(scale_factors, 0);
  c10::SmallVector<int64_t, SIZE> out_size = upsample_linear1d_npu_output_size(input, compute_size, false, scales_w);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(input, out_size);

  auto result_size = at::IntArrayRef(compute_size);
  EXEC_NPU_CMD(aclnnUpsampleNearest1d, input, result_size, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::upsample_nearest1d(const at::Tensor& self,
                                                       at::IntArrayRef output_size,
                                                       c10::optional<double> scales) {
  DO_COMPATIBILITY(aclnnUpsampleNearest1d, NPUNativeFunctions::upsample_nearest1d(self, output_size, scales));
  c10::SmallVector<int64_t, SIZE> out_size = upsample_linear1d_npu_output_size(self, output_size, false, scales);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, out_size);
  
  EXEC_NPU_CMD(aclnnUpsampleNearest1d, self, output_size, result);
  return result;
}
} // namespace native
} // namespace at_npu
