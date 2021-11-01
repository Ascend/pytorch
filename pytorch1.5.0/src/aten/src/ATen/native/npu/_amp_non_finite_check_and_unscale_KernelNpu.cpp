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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

void _amp_non_finite_check_and_unscale_out_npu_(
    Tensor& self, Tensor& found_inf, const Tensor& inv_scale) {
  TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");
  TORCH_CHECK(self.layout() == at::kStrided, "self must be a strided (not sparse) Tensor.");

  // The nan and INF judgments are left alone, and found_inf is set to 0.0 by default
  found_inf[0] = 0.0;
  // CalcuOpUtil::execute_npu_operate("Mul", inputs, outputs, attrs);
  OpCommand cmd;
  cmd.Name("Mul")
      .Input(self)
      .Input(inv_scale)
      .Output(self)
      .Run();
}

void _amp_non_finite_check_and_unscale_npu_(
    Tensor& self, Tensor& found_inf, const Tensor& inv_scale) {
  _amp_non_finite_check_and_unscale_out_npu_(self, found_inf, inv_scale);
}
} // namespace native
} // namespace at