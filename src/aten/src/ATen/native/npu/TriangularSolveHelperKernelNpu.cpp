// Copyright (c) 2021 Huawei Technologies Co., Ltd
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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

std::tuple<Tensor, Tensor> _triangular_solve_helper_npu(
    const Tensor& self,
    const Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular) {
  TORCH_CHECK(self.dtype() == at::kFloat && A.dtype() == at::kFloat,
        "_triangular_solve_helper_npu only supported Float, but get ", self.dtype(), ' ', A.dtype());
  auto self_working_copy = OpPreparation::ApplyTensor(self);
  auto A_working_copy = A.clone();

  Tensor A_tensor = A;
  if (unitriangular) {
    auto diagonal_tensor = at::eye(A_tensor.size(-2), A_tensor.size(-1), A_tensor.options());
    A_tensor = A_tensor * (1 - diagonal_tensor) + diagonal_tensor;
  }
  OpCommand cmd;
  cmd.Name("MatrixTriangularSolve")
    .Input(A_tensor)
    .Input(self)
    .Output(self_working_copy)
    .Attr("lower", !upper)
    .Attr("adjoint", transpose)
    .Run();
  return std::tuple<Tensor, Tensor>(self_working_copy, A_working_copy);
}
}
}
