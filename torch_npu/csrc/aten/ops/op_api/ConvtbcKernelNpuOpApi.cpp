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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::conv_tbc(const at::Tensor &self, const at::Tensor &weight, const at::Tensor &bias,
                                             int64_t pad) {
  DO_COMPATIBILITY(aclnnConvTbc, NPUNativeFunctions::conv_tbc(self, weight, bias, pad));
  int64_t Wo = self.size(0) + 2 * pad - weight.size(0) + 1;
  c10::SmallVector<int64_t, SIZE> outputSize = {Wo, self.size(1), weight.size(2)};
  at::Tensor output = OpPreparation::ApplyTensorWithoutFormat(self, outputSize);
  int8_t cube_math_type = 1;
  EXEC_NPU_CMD(aclnnConvTbc, self, weight, bias, pad, output, cube_math_type);
  return output;
}
}  // namespace native
}  // namespace at_npu
