// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::embedding_backward_symint(
    const at::Tensor& grad, 
    const at::Tensor& indices, 
    c10::SymInt num_weights,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq, 
    bool sparse) {
    TORCH_CHECK(sparse == false, "NPU error, not yet support sparse tensor, when sparse == True");

    // run dense tensor backward
    return at::embedding_dense_backward(
        grad,
        indices,
        num_weights.guard_int(__FILE__, __LINE__),
        padding_idx.guard_int(__FILE__, __LINE__),
        scale_grad_by_freq);
}

} // namespace native
} // namespace at