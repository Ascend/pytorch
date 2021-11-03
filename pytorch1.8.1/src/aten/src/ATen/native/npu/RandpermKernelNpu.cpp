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

Tensor& randperm_generator_out_npu(int64_t n, optional<Generator> generator, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);

  OpCommand cmd;
  cmd.Name("Randperm")
       .Output(result)
       .Run();

  return result;
}

Tensor& randperm_out_npu(int64_t n, Tensor& result) {
  return randperm_generator_out_npu(n, static_cast<optional<Generator>>(c10::nullopt), result);
}

Tensor randperm_generator_npu(
    int64_t n,
    optional<Generator> generator,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory) {
  TensorOptions options;
  options = options.dtype(dtype)
                   .layout(layout)
                   .device(device);

  Tensor result = OpPreparation::ApplyTensorWithFormat(
      {n},
      options,
      ACL_FORMAT_NCHW);

  return randperm_generator_out_npu(n, generator, result);
}

Tensor randperm_npu(
    int64_t n,
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device,
    optional<bool> pin_memory) {
  return randperm_generator_npu(n, static_cast<optional<Generator>>(nullopt), dtype, layout, device, pin_memory);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("randperm", TORCH_FN(randperm_npu));
  m.impl("randperm.generator", TORCH_FN(randperm_generator_npu));
  m.impl("randperm.out", TORCH_FN(randperm_out_npu));
  m.impl("randperm.generator_out", TORCH_FN(randperm_generator_out_npu));
}
} // namespace native
} // namespace at
