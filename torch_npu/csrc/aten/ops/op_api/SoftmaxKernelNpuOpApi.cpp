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
at::Tensor NPUNativeOpApiFunctions::_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  DO_COMPATIBILITY(aclnnSoftmax, NPUNativeFunctions::_softmax(self, dim, half_to_float));
  // construct the output tensor of the NPU
  at::Tensor result;
  if (half_to_float) {
    result = OpPreparation::ApplyTensor(self, self.options().dtype(at::ScalarType::Float));
  } else {
    result = OpPreparation::ApplyTensor(self);
  }

  EXEC_NPU_CMD(aclnnSoftmax, self, dim, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::_softmax_out(const at::Tensor& self, int64_t dim, bool half_to_float,
                                                  at::Tensor& out) {
  DO_COMPATIBILITY(aclnnSoftmax, NPUNativeFunctions::_softmax_out(self, dim, half_to_float, out));
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnSoftmax, self, dim, out);
  return out;
}
}  // namespace native
}  // namespace at_npu
