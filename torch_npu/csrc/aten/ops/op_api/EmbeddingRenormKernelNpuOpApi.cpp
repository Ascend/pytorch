// Copyright (c) 2020-2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::embedding_renorm_(at::Tensor& self, const at::Tensor& indices, double max_norm,
                                                       double norm_type) {
  DO_COMPATIBILITY(aclnnEmbeddingRenorm, NPUNativeFunctions::embedding_renorm_(self, indices, max_norm, norm_type));
  auto self_arg = at::TensorArg(self, "self", 1);
  auto indices_arg = at::TensorArg(indices, "indices", 2);
  at::checkDim("embedding_renorm_", self_arg, 2);
  at::checkScalarTypes("embedding_renorm_", indices_arg, {at::kLong, at::kInt});

  at::Tensor indices_copy = indices.clone();
  auto num_indices = indices.numel();
  at::native::resize_(indices_copy, num_indices);

  EXEC_NPU_CMD(aclnnEmbeddingRenorm, self, indices_copy, max_norm, norm_type);

  return self;
}

}  // namespace native
}  // namespace at_npu
