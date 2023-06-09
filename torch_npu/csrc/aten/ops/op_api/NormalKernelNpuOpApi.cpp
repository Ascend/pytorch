// Copyright (c) 2022, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::normal_(at::Tensor& self, double mean, double std,
                                             c10::optional<at::Generator> generator) {
  DO_COMPATIBILITY(aclnnInplaceNormal, NPUNativeFunctions::normal_(self, mean, std, generator));
  TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std);
  OpPreparation::CheckOut({}, self, self, self.sizes());
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  float mean_cast = static_cast<float>(mean);
  float rstd_cast = static_cast<float>(std);
  EXEC_NPU_CMD(aclnnInplaceNormal, self, mean_cast, rstd_cast, seed, offset);
  return self;
}

}  // namespace native
}  // namespace at_npu
