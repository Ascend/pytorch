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
#include <ATen/native/LinearAlgebraUtils.h>

namespace at_npu {
namespace native {
std::tuple<at::Tensor, at::Tensor> npu_triangular_solve_helper(
    const at::Tensor& self,
    const at::Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular) {

  at::Tensor self_broadcasted, A_broadcasted;
  std::tie(self_broadcasted, A_broadcasted) = at::native::_linalg_broadcast_batch_dims(self, A, "triangular_solve");
  TORCH_CHECK(self_broadcasted.dtype() == at::kFloat && A_broadcasted.dtype() == at::kFloat,
      "_triangular_solve_helper_npu only supported Float, but get ", self_broadcasted.dtype(), ' ', A_broadcasted.dtype());
  auto self_working_copy = OpPreparation::ApplyTensor(self_broadcasted);
  auto A_working_copy = A_broadcasted.clone();
  at::Tensor A_tensor = A_broadcasted;
  if (unitriangular) {
    auto diagonal_tensor = at::eye(A_tensor.size(-2), A_tensor.size(-1), A_tensor.options());
    A_tensor = A_tensor * (1 - diagonal_tensor) + diagonal_tensor;
  }
  OpCommand cmd;
  cmd.Name("MatrixTriangularSolve")
    .Input(A_tensor)
    .Input(self_broadcasted)
    .Output(self_working_copy)
    .Attr("lower", !upper)
    .Attr("adjoint", transpose)
    .Run();

  return std::tuple<at::Tensor, at::Tensor>(self_working_copy, A_working_copy);
}

std::tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::triangular_solve_out(const at::Tensor& self, const at::Tensor& A, bool upper,
                                                                              bool transpose, bool unitriangular, at::Tensor& result,
                                                                              at::Tensor& clone_A) {
  at::Tensor result_tmp, clone_A_tmp;
  std::tie(result_tmp, clone_A_tmp) = npu_triangular_solve_helper(self, A, upper, transpose, unitriangular);
  result.resize_as_(result_tmp).copy_(result_tmp);
  clone_A.resize_as_(clone_A_tmp).copy_(clone_A_tmp);
  return std::tuple<at::Tensor&, at::Tensor&>(result, clone_A);
}
} // namespace native
} // namespace at_npu
