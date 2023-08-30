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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::isin_out(const at::Tensor& elements,
                                              const at::Scalar& test_elements,
                                              bool assume_unique,
                                              bool invert,
                                              at::Tensor& result) {
  DO_COMPATIBILITY(aclnnIsInTensorScalar, NPUNativeFunctions::isin_out(elements, test_elements, assume_unique, invert,
      result));
  OpPreparation::CheckOut({elements}, result, at::ScalarType::Bool, elements.sizes());
  EXEC_NPU_CMD(aclnnIsInTensorScalar, elements, test_elements, assume_unique, invert, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::isin(const at::Tensor& elements,
                                         const at::Scalar& test_elements,
                                         bool assume_unique,
                                         bool invert) {
  DO_COMPATIBILITY(aclnnIsInTensorScalar, NPUNativeFunctions::isin(elements, test_elements, assume_unique, invert));

  at::Tensor result =
      OpPreparation::ApplyTensorWithoutFormat(elements.sizes(), elements.options().dtype(at::kBool));

  EXEC_NPU_CMD(aclnnIsInTensorScalar, elements, test_elements, assume_unique, invert, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
