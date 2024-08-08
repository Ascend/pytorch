
// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::clone(const at::Tensor &src, c10::optional<c10::MemoryFormat> format)
{
  DO_COMPATIBILITY(aclnnInplaceCopy, NPUNativeFunctions::clone(src, format));
  auto baseSelf = OpPreparation::ApplyTensorWithoutFormat(src);
  baseSelf.copy_(src);
  return baseSelf;
}

at::Tensor NPUNativeOpApiFunctions::scalar_tensor(const c10::Scalar& s, c10::optional<at::ScalarType> dtype,
                                                  c10::optional<at::Layout> layout, c10::optional<at::Device> device,
                                                  c10::optional<bool> pin_memory) {
  DO_COMPATIBILITY(aclnnInplaceFillScalar, NPUNativeFunctions::scalar_tensor(s, dtype, layout, device, pin_memory));
  at::tracer::impl::NoTracerDispatchMode tracer_guard;
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  auto result = at::native::empty_cpu({}, dtype, layout, c10::make_optional(c10::Device(at::kCPU)), pin_memory);
  auto result_npu = result.to(at::device(at_npu::key::NativeDeviceType));
  EXEC_NPU_CMD(aclnnInplaceFillScalar, result_npu, s);
  return result_npu;
}
}  // namespace native
}  // namespace at_npu

