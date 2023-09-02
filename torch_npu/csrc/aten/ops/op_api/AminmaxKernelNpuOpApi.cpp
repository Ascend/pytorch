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

std::tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::_aminmax(const at::Tensor &self,
                                                                    const int64_t dim,
                                                                    bool keepdim) {
  DO_COMPATIBILITY(aclnnAminmaxDim, NPUNativeFunctions::_aminmax(self, dim, keepdim));
  at::SmallVector<int64_t, SIZE> dims = {dim};
  auto output_size = reduce_ops_npu_output_size(self, dims, keepdim);
  auto min = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  auto max = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  EXEC_NPU_CMD(aclnnAminmaxDim, self, dim, keepdim, min, max);
  return std::tie(min, max);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::_aminmax(const at::Tensor &self) {
  DO_COMPATIBILITY(aclnnAminmaxDim, NPUNativeFunctions::_aminmax(self));
  at::IntArrayRef dims = CalcuOpUtil::GetDimlistForTensor(self);
  auto output_size = reduce_ops_npu_output_size(self, dims, false);
  auto min = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  auto max = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  EXEC_NPU_CMD(aclnnAminmaxAll, self, min, max);
  return std::tie(min, max);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::aminmax(const at::Tensor &self,
                                                                    c10::optional<int64_t> dim,
                                                                    bool keepdim) {
  DO_COMPATIBILITY(aclnnAminmax, NPUNativeFunctions::aminmax(self, dim, keepdim));
  at::IntArrayRef dims;
  if (dim.has_value()) {
    dims = dim.value();
  } else {
    dims = CalcuOpUtil::GetDimlistForTensor(self);
  }
  auto output_size = reduce_ops_npu_output_size(self, dims, keepdim);
  auto min = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  auto max = OpPreparation::ApplyTensorWithoutFormat(self, output_size);
  EXEC_NPU_CMD(aclnnAminmax, self, dims, keepdim, min, max);
  return std::tie(min, max);
}

std::tuple<at::Tensor &, at::Tensor &> NPUNativeOpApiFunctions::aminmax_out(const at::Tensor &self,
                                                                            c10::optional<int64_t> dim,
                                                                            bool keepdim,
                                                                            at::Tensor &min,
                                                                            at::Tensor &max) {
  DO_COMPATIBILITY(aclnnAminmax, NPUNativeFunctions::aminmax_out(self, dim, keepdim, min, max));
  at::IntArrayRef dims;
  if (dim.has_value()) {
    dims = dim.value();
  } else {
    dims = CalcuOpUtil::GetDimlistForTensor(self);
  }
  auto output_size = reduce_ops_npu_output_size(self, dims, keepdim);
  OpPreparation::CheckOut({self}, min, self.scalar_type(), output_size);
  OpPreparation::CheckOut({self}, max, self.scalar_type(), output_size);
  EXEC_NPU_CMD(aclnnAminmax, self, dims, keepdim, min, max);
  return std::tie(min, max);
}

} // namespace native
} // namespace at_npu
