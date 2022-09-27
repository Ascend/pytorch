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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

constexpr float_t EPSILON = 1e-5;

namespace at_npu {
namespace native {

static inline void normalize_batch_check(
    const at::Tensor& self,
    const at::Tensor& seq_len,
    int64_t normalize_type){
  TORCH_CHECK(
      seq_len.dim() == 1,
      "Non-empty 1D seq_len tensor expected but got a tensor with sizes ",
      seq_len.sizes());
  TORCH_CHECK(
      seq_len.size(0) == self.size(0),
      "seq_len's length should be equal self' num, but got seq_len length ",
      seq_len.size(0),
      "self num ",
      self.size(0));
  TORCH_CHECK(
      normalize_type >= 0 && normalize_type <= 1,
      "normalize_type expected to be in range [0, 1], but got ",
      normalize_type);
}

at::Tensor NPUNativeFunctions::npu_normalize_batch(
    const at::Tensor& self,
    const at::Tensor& seq_len,
    int64_t normalize_type){
  normalize_batch_check(self, seq_len, normalize_type);
  // apply output tensor
  at::Tensor result = OpPreparation::ApplyTensor(self);
  string normalizeType = normalize_type == 0 ? "per_feature" : "all_features";

  OpCommand cmd;
  cmd.Name("NormalizeBatch")
      .Input(self)
      .Input(seq_len)
      .Output(result)
      .Attr("normalize_type", normalizeType)
      .Attr("epsilon", EPSILON)
      .Run();
  return result;
}

} // namespace native
} // namespace at