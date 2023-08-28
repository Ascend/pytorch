// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"

namespace at_npu {
namespace native {

static const int64_t MIN_DEPTH = 1;
static const int64_t AUTO_DEPTH = -1;
static const int64_t MIN_NUM_CLASSES = 0;

at::Tensor NPUNativeOpApiFunctions::one_hot(const at::Tensor& self, int64_t num_classes) {
  DO_COMPATIBILITY(aclnnOneHot, NPUNativeFunctions::one_hot(self, num_classes));

  int64_t depth = num_classes;

  TORCH_CHECK(depth >= AUTO_DEPTH, "NPU error, not yet support negative num_classes, when num_classes less than -1");

  // when the self is empty, num_classes should be greater than 0
  TORCH_CHECK(self.numel() != 0 || num_classes > MIN_NUM_CLASSES,
              "NPU error, can not infer total number of classes from empty tensor.");

  if (depth == AUTO_DEPTH) {
    depth = self.max().item().toLong() + 1;
    if (depth < MIN_DEPTH) {
      depth = MIN_DEPTH;
    }
  }

  // construct on_value tensor
  at::Tensor on_value_tensor = OpPreparation::ApplyTensorWithoutFormat({1}, self.options());
  on_value_tensor.fill_(1);

  // construct off_value tensor
  at::Tensor off_value_tensor = OpPreparation::ApplyTensorWithoutFormat({1}, self.options());
  off_value_tensor.fill_(0);

  auto output_size = array_to_small_vector(self.sizes());
  output_size.emplace_back(depth);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(output_size, self.options(), self);

  int64_t axis = -1;
  EXEC_NPU_CMD(aclnnOneHot, self, depth, on_value_tensor, off_value_tensor, axis, result);
  return result;
}
}  // namespace native
}  // namespace at_npu
