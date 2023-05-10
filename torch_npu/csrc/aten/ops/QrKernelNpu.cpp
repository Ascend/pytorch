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

namespace at_npu {
namespace native {

std::tuple<c10::SmallVector<int64_t, N>, c10::SmallVector<int64_t, N>> qr_npu_output_size(
    const at::Tensor& self,
    bool some) {
  int m = self.size(-2);
  int n = self.size(-1);
  auto k = std::min<int>(m, n);
  auto shape = array_to_small_vector(self.sizes());
  c10::SmallVector<int64_t, N> Qsize(shape.begin(), shape.end()-2);
  c10::SmallVector<int64_t, N> Rsize(shape.begin(), shape.end()-2);
  if(some){
    Qsize.insert(Qsize.end(), {m, k});
    Rsize.insert(Rsize.end(), {k, n});
  } else {
    Qsize.insert(Qsize.end(), {m, m});
    Rsize.insert(Rsize.end(), {m, n});
  }
  return std::tie(Qsize, Rsize);
}

static inline void qr_check(
    const at::Tensor& self) {
  TORCH_CHECK(
      self.ndimension() >= 2,
      "Expected nonempty least 2D tensor, but got a tensor with sizes ",
      self.dim());
}

std::tuple<at::Tensor&, at::Tensor&> qr_out_npu_nocheck(
    at::Tensor& Q,
    at::Tensor& R,
    const at::Tensor& self,
    bool some) {
  bool full_matrices = !some;
  OpCommand cmd;
  cmd.Name("Qr")
      .Input(self)
      .Output(Q)
      .Output(R)
      .Attr("full_matrices", full_matrices)
      .Run();
  return std::tie(Q, R);
}

std::tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::qr_out(
    const at::Tensor& self,
    bool some,
    at::Tensor& Q,
    at::Tensor& R) {
  qr_check(self);
  auto sizes = qr_npu_output_size(self, some);
  OpPreparation::CheckOut(
      {self},
      Q,
      self,
      std::get<0>(sizes));
  OpPreparation::CheckOut(
      {self},
      R,
      self,
      std::get<1>(sizes));
  return qr_out_npu_nocheck(Q, R, self, some);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeFunctions::qr(
    const at::Tensor& self,
    bool some) {
  qr_check(self);
  auto sizes = qr_npu_output_size(self, some);
  at::Tensor Q = OpPreparation::ApplyTensor(self, std::get<0>(sizes));
  at::Tensor R = OpPreparation::ApplyTensor(self, std::get<1>(sizes));

  qr_out(self, some, Q, R);
  return std::tie(Q, R);
}

} // namespace native
} // namespace at_npu