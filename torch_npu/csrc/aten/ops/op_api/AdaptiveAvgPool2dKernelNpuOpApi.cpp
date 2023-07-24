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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::adaptive_avg_pool2d_out(const at::Tensor& self, at::IntArrayRef output_size,
                                                             at::Tensor& result) {
  DO_COMPATIBILITY(aclnnAdaptiveAvgPool2d, NPUNativeFunctions::adaptive_avg_pool2d_out(self, output_size, result));
  EXEC_NPU_CMD(aclnnAdaptiveAvgPool2d, self, output_size, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size) {
  DO_COMPATIBILITY(aclnnAdaptiveAvgPool2d, NPUNativeFunctions::adaptive_avg_pool2d(self, output_size));
  // The logic is a little different from CPU_impl
  // can't use "NPUNativeOpApiFunctions::_adaptive_avg_pool2d(self, output_size)", this will resnet50 accuracy error
  return at::_adaptive_avg_pool2d(self, output_size);
}

at::Tensor NPUNativeOpApiFunctions::_adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size) {
  DO_COMPATIBILITY(aclnnAdaptiveAvgPool2d, NPUNativeFunctions::_adaptive_avg_pool2d(self, output_size));

  TORCH_CHECK((self.dim() == 3 || self.dim() == 4), "non-empty 3D or 4D (batch mode) tensor expected for input");
  auto outputSize = array_to_small_vector(self.sizes());
  outputSize[self.dim() - 1] = output_size[1];
  outputSize[self.dim() - 2] = output_size[0];

  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, outputSize);

  NPUNativeOpApiFunctions::adaptive_avg_pool2d_out(self, output_size, result);

  return result;
}

}  // namespace native
}  // namespace at_npu
