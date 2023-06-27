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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor &argmin_exec(
    const at::Tensor& self, 
    at::optional<int64_t> dim, 
    bool keepdim,
    at::Tensor& result, bool out_mode) {
  TORCH_CHECK(!(self.numel()==0 && !(dim.has_value())), "Expected reduction dim to be specified for input.numl()==0")
  at::Tensor input;
  int64_t real_dim = 0;
  bool real_keep_dim = false;
  if (dim.has_value()) {
    input = self;
    real_dim = dim.value();
    real_keep_dim = keepdim;
  } else {
    input = self.reshape({-1});
  }

  // calculate the output size
  auto output_size = reduce_ops_npu_output_size(input, real_dim, real_keep_dim);

  if (out_mode) {
    OpPreparation::CheckOut({self}, result, result, output_size);
  } else {
    // construct the output tensor of the NPU
    result = OpPreparation::ApplyTensorWithSizes(output_size, self.options().dtype(at::kLong));
  }

  EXEC_NPU_CMD(aclnnArgMin, input, real_dim, real_keep_dim, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::argmin(const at::Tensor& self, at::optional<int64_t> dim, bool keepdim) {
  DO_COMPATIBILITY(aclnnArgMin, NPUNativeFunctions::argmin(self, dim, keepdim));

  at::Tensor result;
  return argmin_exec(self, dim, keepdim, result, false);
}

at::Tensor &NPUNativeOpApiFunctions::argmin_out(
    const at::Tensor& self, 
    at::optional<int64_t> dim, 
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnArgMin, NPUNativeFunctions::argmin_out(self, dim, keepdim, result));

  return argmin_exec(self, dim, keepdim, result, true);
}

} // namespace native
} // namespace at_npu
