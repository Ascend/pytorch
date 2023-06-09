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

#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::put_(at::Tensor& self, const at::Tensor& index, const at::Tensor& source,
                                          bool accumulate) {
  DO_COMPATIBILITY(aclnnInplacePut, NPUNativeFunctions::put_(self, index, source, accumulate));

  c10::SmallVector<at::Tensor, N> inputs = {self, index, source};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

  EXEC_NPU_CMD(aclnnInplacePut, self, index, source, accumulate);
  return self;
}
}  // namespace native
}  // namespace at_npu