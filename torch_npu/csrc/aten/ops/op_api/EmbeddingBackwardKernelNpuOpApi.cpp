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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::embedding_backward(
    const at::Tensor& grad, 
    const at::Tensor& indices, 
    int64_t num_weights, 
    int64_t padding_idx, 
    bool scale_grad_by_freq, 
    bool sparse) {
  DO_COMPATIBILITY(aclnnEmbeddingDenseBackward, NPUNativeFunctions::embedding_backward(grad, indices, num_weights,
                                                                                       padding_idx, scale_grad_by_freq,
                                                                                       sparse));
  TORCH_CHECK(sparse == false, "NPU error, not yet support sparse tensor, when sparse == True");
  // run dense tensor backward
  return NPUNativeOpApiFunctions::embedding_dense_backward(
      grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}

} // namespace native
} // namespace at_npu

