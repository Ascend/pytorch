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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::argmax(const at::Tensor& self, at::optional<int64_t> dim, bool keepdim) {
  DO_COMPATIBILITY(aclnnArgMax, NPUNativeFunctions::argmax(self, dim, keepdim));

  if (self.numel() == 0) {
    return self;
  }

  at::Tensor input = self.reshape({-1});
  int64_t realDim = 0;
  bool realKeepDim = false;
  if (dim.has_value()) {
    input = self;
    realDim = dim.value();
    realKeepDim = keepdim;
  }

  // calculate the output size
  auto outputSize = reduce_ops_npu_output_size(input, realDim, realKeepDim);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithSizes(outputSize, self.options().dtype(at::kInt));

  EXEC_NPU_CMD(aclnnArgMax, input, realDim, realKeepDim, result);
  return result;
}

} // namespace native
} // namespace at_npu
