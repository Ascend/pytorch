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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;
const int FLOAT_STATUS_OP_DIMS_SIZE = 8;

Tensor alloc_float_status_npu(const Tensor& self) {
  auto options = at::TensorOptions(at::kNPU).dtype(at::kFloat);
  Tensor result = at::empty({FLOAT_STATUS_OP_DIMS_SIZE}, options);
  OpCommand cmd;
  cmd.Name("NPUAllocFloatStatus")
      .Output(result)
      .Run();

  return result;
}

Tensor get_float_status_npu(const Tensor& self) {
  Tensor result = at::empty({FLOAT_STATUS_OP_DIMS_SIZE}, self.options());
  OpCommand cmd;
  cmd.Name("NPUGetFloatStatus")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

Tensor clear_float_status_npu(const Tensor& self) {
  Tensor result = at::empty({FLOAT_STATUS_OP_DIMS_SIZE}, self.options());
  OpCommand cmd;
  cmd.Name("NPUClearFloatStatus")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("npu_alloc_float_status", TORCH_FN(alloc_float_status_npu));
  m.impl("npu_get_float_status", TORCH_FN(get_float_status_npu));
  m.impl("npu_clear_float_status", TORCH_FN(clear_float_status_npu));
}
} // namespace native
} // namespace at