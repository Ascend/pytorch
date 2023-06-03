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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::fill_(at::Tensor& self, const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnInplaceFillScalar, NPUNativeFunctions::fill_(self, value));
  EXEC_NPU_CMD(aclnnInplaceFillScalar, self, value);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::fill_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceFillScalar, NPUNativeFunctions::fill_(self, other));
  DO_COMPATIBILITY(aclnnInplaceFillTensor, NPUNativeFunctions::fill_(self, other));
  if (other.dim() == 0 && !at_npu::key::isDeviceTensor(other)) {
    const at::Scalar other_value = other.item();
    EXEC_NPU_CMD(aclnnInplaceFillScalar, self, other_value);
  } else {
    EXEC_NPU_CMD(aclnnInplaceFillTensor, self, other);
  }
  return self;
}

}  // namespace native
}  // namespace at_npu
