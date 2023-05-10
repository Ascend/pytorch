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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::cumsum_out(const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor& result) {
 // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  TORCH_CHECK(
      !dtype.has_value() || (result.scalar_type() == dtype.value()),
      "provided dtype must match dtype of result in cumsum. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  {
    at::NoNamesGuard guard;
    OpCommand cmd;
    // if dim = 0, performance in Aicpu is better than Aicore
    // if dim > INT32_MAX, we should use long to store dim for ensuring function correctness.
    // use host memory instead of scalar to improve delivery performance
    at::Scalar dimScalar(dim);
    cmd.Name("Cumsum")
        .Input(self.toType(result.scalar_type()));
    if (dim == 0 || dim > INT32_MAX) {
        cmd.Input(dimScalar, at::kLong, CompileType::MEMORY_HOST_COMPILE_DEPENDENT);
    } else {
        cmd.Input(dimScalar, at::kInt, CompileType::MEMORY_HOST_COMPILE_DEPENDENT);
    }
    cmd.Output(result)
        .Run();
  }
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor& NPUNativeFunctions::cumsum_out(const at::Tensor& self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor& result) {
  return NPUNativeFunctions::cumsum_out(self, dimname_to_position(self, dim), dtype, result);
}

} // namespace native
} // namespace at_npu