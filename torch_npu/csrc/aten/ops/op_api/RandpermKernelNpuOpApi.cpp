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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>

namespace at_npu {
namespace native {

at::Tensor& randperm_op_api(int64_t n, c10::optional<at::Generator> gen_, at::Tensor& result) {
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(gen_, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  EXEC_NPU_CMD(aclnnRandperm, n, pair.first, pair.second, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::randperm_out(int64_t n, c10::optional<at::Generator> generator, at::Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  OpPreparation::CheckOut({}, result, result, {n});
  randperm_op_api(n, generator, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::randperm_out(int64_t n, at::Tensor& result) {
  OpPreparation::CheckOut({}, result, result, {n});
  c10::optional<at::Generator> generator = static_cast<c10::optional<at::Generator>>(c10::nullopt);
  randperm_op_api(n, generator, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::randperm(
    int64_t n,
    c10::optional<at::Generator> generator,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  at::TensorOptions options;
  options = options.dtype(dtype)
                   .layout(layout)
                   .device(device)
                   .pinned_memory(pin_memory);

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      {n},
      options,
      ACL_FORMAT_ND);

  randperm_op_api(n, generator, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::randperm(
    int64_t n,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  return NPUNativeOpApiFunctions::randperm(n, static_cast<c10::optional<at::Generator>>(c10::nullopt), dtype, layout, device, pin_memory);
}
} // namespace native
} // namespace at_npu
