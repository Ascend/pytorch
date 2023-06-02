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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::masked_fill_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value) {
  // OpPreparation::CheckMemory({self, mask, value}, {self});
  DO_COMPATIBILITY(aclnnInplaceMaskedFillTensor, NPUNativeFunctions::masked_fill_(self, mask, value));
  EXEC_NPU_CMD(aclnnInplaceMaskedFillTensor, self, mask, value);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::masked_fill_(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnInplaceMaskedFillScalar, NPUNativeFunctions::masked_fill_(self, mask, value));
  EXEC_NPU_CMD(aclnnInplaceMaskedFillScalar, self, mask, value);
  return self;
}

}  // namespace native
}  // namespace at_npu
