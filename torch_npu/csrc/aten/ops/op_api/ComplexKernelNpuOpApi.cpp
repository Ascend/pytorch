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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::complex_out(const at::Tensor& real, const at::Tensor& imag,
                                                 at::Tensor& result) {
  DO_COMPATIBILITY(aclnnComplex, NPUNativeFunctions::complex_out(real, imag, result));
  auto outputSize = broadcast_ops_npu_output_size(real, imag);
  OpPreparation::CheckOut({real}, result, result.scalar_type(), outputSize);
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnComplex, real, imag, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::complex(const at::Tensor& real, const at::Tensor& imag) {
  DO_COMPATIBILITY(aclnnComplex, NPUNativeFunctions::complex(real, imag));
  at::ScalarType high_type = at::native::result_type(real, imag);
  if (high_type == at::ScalarType::Float) {
    high_type = at::ScalarType::ComplexFloat;
  } else if (high_type == at::ScalarType::Double) {
    high_type = at::ScalarType::ComplexDouble;
  } else if (high_type == at::ScalarType::Half) {
    high_type = at::ScalarType::ComplexHalf;
  }
  auto outputSize = broadcast_ops_npu_output_size(real, imag);
  at::Tensor result = 
    OpPreparation::ApplyTensorWithoutFormat(outputSize, real.options().dtype(high_type));
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnComplex, real, imag, result);
  return result;
}

} // namespace native
} // namespace at_npu
