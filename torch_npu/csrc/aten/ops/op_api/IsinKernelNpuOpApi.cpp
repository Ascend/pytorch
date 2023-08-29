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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::isin_out(const at::Scalar& element, const at::Tensor &test_element,
                                              bool assume_unique, bool invert, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnIsinScalarTensor,
                   NPUNativeFunctions::isin_out(element, test_element, assume_unique, invert, result));
  c10::SmallVector<int64_t, SIZE> shape_small_vec;
  OpPreparation::CheckOut({test_element}, result, at::ScalarType::Bool, shape_small_vec);
  EXEC_NPU_CMD(aclnnIsinScalarTensor, element, test_element, assume_unique, invert, result);
  return result;
}
} // namespace native
} // namespace at_npu
