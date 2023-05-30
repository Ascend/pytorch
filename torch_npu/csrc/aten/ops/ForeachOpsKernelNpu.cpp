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

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {
#define FOREACH_BINARY_OP_LIST_ALPHA_NPU(OP)                                                                          \
std::vector<at::Tensor> NPUNativeFunctions::_foreach_##OP(at::TensorList tensors1,                                    \
    at::TensorList tensors2, const at::Scalar& alpha) {                                                               \
  return at::native::foreach_tensor_##OP##_list_kernel_slow(tensors1, tensors2, alpha);                               \
}                                                                                                                     \
                                                                                                                      \
void NPUNativeFunctions::_foreach_##OP##_(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar& alpha) {\
  at::native::foreach_tensor_##OP##_list_kernel_slow_(tensors1, tensors2, alpha);                                     \
}

#define FOREACH_BINARY_OP_SCALAR_NPU(OP)                                                                            \
void NPUNativeFunctions::_foreach_##OP##_(at::TensorList tensors, const at::Scalar& scalar) {                       \
  at::native::foreach_tensor_##OP##_scalar_kernel_slow_(tensors, scalar);                                           \
}                                                                                                                   \
                                                                                                                    \
std::vector<at::Tensor> NPUNativeFunctions::_foreach_##OP(at::TensorList tensors, const at::Scalar& scalar) {       \
  return at::native::foreach_tensor_##OP##_scalar_kernel_slow(tensors, scalar);                                     \
}

#define FOREACH_BINARY_OP_SCALARLIST_NPU(OP)                                                                          \
void NPUNativeFunctions::_foreach_##OP##_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {                 \
  at::native::foreach_tensor_##OP##_scalarlist_kernel_slow_(tensors, scalars);                                        \
}                                                                                                                     \
                                                                                                                      \
std::vector<at::Tensor> NPUNativeFunctions::_foreach_##OP(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) { \
  return at::native::foreach_tensor_##OP##_scalarlist_kernel_slow(tensors, scalars);                                  \
}

#define FOREACH_BINARY_OP_LIST_NPU(OP)                                                                               \
std::vector<at::Tensor> NPUNativeFunctions::_foreach_##OP(at::TensorList tensors1, at::TensorList tensors2) {        \
  return at::native::foreach_tensor_##OP##_list_kernel_slow(tensors1, tensors2);                                     \
}                                                                                                                    \
                                                                                                                     \
void NPUNativeFunctions::_foreach_##OP##_(at::TensorList tensors1, at::TensorList tensors2) {                        \
  at::native::foreach_tensor_##OP##_list_kernel_slow_(tensors1, tensors2);                                           \
}

#define FOREACH_UNARY_OP_NPU(OP)                                               \
std::vector<at::Tensor> NPUNativeFunctions::_foreach_##OP(at::TensorList tensors) {       \
  return at::native::foreach_tensor_##OP##_slow(tensors);                                 \
}                                                                                         \
                                                                                          \
void NPUNativeFunctions::_foreach_##OP##_(at::TensorList tensors) {                       \
  at::native::foreach_tensor_##OP##_slow_(tensors);                                       \
}

#define FOREACH_POINTWISE_OP_SCALAR_NPU(OP)                                                                           \
std::vector<at::Tensor> NPUNativeFunctions::_foreach_##OP(at::TensorList input, at::TensorList tensors1,              \
    at::TensorList tensors2, const at::Scalar& scalar) {                                                              \
  return at::native::foreach_tensor_##OP##_scalar_slow(input, tensors1, tensors2, scalar);                            \
}                                                                                                                     \
                                                                                                                      \
void NPUNativeFunctions::_foreach_##OP##_(at::TensorList input, at::TensorList tensors1,                              \
    at::TensorList tensors2, const at::Scalar& scalar) {                                                              \
  at::native::foreach_tensor_##OP##_scalar_slow_(input, tensors1, tensors2, scalar);                                  \
}

#define FOREACH_POINTWISE_OP_SCALARLIST_NPU(OP)                                                                       \
std::vector<at::Tensor> NPUNativeFunctions::_foreach_##OP(at::TensorList input, at::TensorList tensors1,              \
    at::TensorList tensors2, at::ArrayRef<at::Scalar> scalars) {                                                      \
  return at::native::foreach_tensor_##OP##_scalarlist_slow(input, tensors1, tensors2, scalars);                       \
}                                                                                                                     \
                                                                                                                      \
void NPUNativeFunctions::_foreach_##OP##_(at::TensorList input, at::TensorList tensors1,                              \
    at::TensorList tensors2, at::ArrayRef<at::Scalar> scalars) {                                                      \
  at::native::foreach_tensor_##OP##_scalarlist_slow_(input, tensors1, tensors2, scalars);                             \
}

FOREACH_BINARY_OP_LIST_ALPHA_NPU(add);

FOREACH_BINARY_OP_SCALAR_NPU(add);
FOREACH_BINARY_OP_SCALAR_NPU(div);
FOREACH_BINARY_OP_SCALAR_NPU(mul);

FOREACH_BINARY_OP_SCALARLIST_NPU(add);
FOREACH_BINARY_OP_SCALARLIST_NPU(mul);
FOREACH_BINARY_OP_SCALARLIST_NPU(div);

FOREACH_BINARY_OP_LIST_NPU(mul);
FOREACH_BINARY_OP_LIST_NPU(div);

FOREACH_UNARY_OP_NPU(sqrt);

FOREACH_POINTWISE_OP_SCALAR_NPU(addcdiv);
FOREACH_POINTWISE_OP_SCALAR_NPU(addcmul);

FOREACH_POINTWISE_OP_SCALARLIST_NPU(addcdiv);
FOREACH_POINTWISE_OP_SCALARLIST_NPU(addcmul);
} // namespace at_npu
} // namespace at_npu
