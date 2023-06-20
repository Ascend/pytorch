// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::_softmax_backward_data_out(const at::Tensor& grad_output,
                                                                const at::Tensor& output, int64_t dim,
                                                                c10::ScalarType input_dtype, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnSoftmaxBackward,
                   NPUNativeFunctions::_softmax_backward_data_out(grad_output, output, dim, input_dtype, result));
  OpPreparation::CheckOut({grad_output, output}, result, grad_output.scalar_type(), grad_output.sizes());
  
  EXEC_NPU_CMD(aclnnSoftmaxBackward, grad_output, output, dim, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::_softmax_backward_data(const at::Tensor& grad_output, const at::Tensor& output,
                                                           int64_t dim, c10::ScalarType input_dtype) {
  DO_COMPATIBILITY(aclnnSoftmaxBackward,
                   NPUNativeFunctions::_softmax_backward_data(grad_output, output, dim, input_dtype));

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensor(output);
  EXEC_NPU_CMD(aclnnSoftmaxBackward, grad_output, output, dim, result);

  return result;
}
}  // namespace native
}  // namespace at_npu
