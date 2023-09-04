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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::_pdist_forward(const at::Tensor& self, double p) {
  DO_COMPATIBILITY(aclnnPdist, NPUNativeFunctions::_pdist_forward(self, p));
  TORCH_CHECK(p >= 0, "pdist only supports non-negative p values");
  // double is not supported in NPU,  type of P needs to be converted from double to float.
  float p_float;
  if (std::isinf(p)) {
    p_float = std::numeric_limits<float>::infinity();
  } else {
    TORCH_CHECK(p <= std::numeric_limits<float>::max(), "npu dose not support float64" );
    p_float = static_cast<float>(p);
  }
  auto output_size = pdist_npu_output_size(self, p_float);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  EXEC_NPU_CMD(aclnnPdist, self, p_float, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::pdist(
    const at::Tensor& self,
    double p) {
  return NPUNativeOpApiFunctions::_pdist_forward(self, p);
}

} // namespace native
} // namespace at_npu
