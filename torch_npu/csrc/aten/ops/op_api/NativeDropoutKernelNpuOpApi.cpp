// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/core/npu/SecondaryStreamGuard.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace at_npu {
namespace native {
static const int64_t BIT_NUMBER = 128;
static const int64_t UINT8_BIT_NUMBER = 8;

static std::tuple<at::Tensor, at::Tensor> _dropout_npu(const at::Tensor& self, double p, bool train) {
  int64_t length = (self.numel() + BIT_NUMBER - 1) / BIT_NUMBER * BIT_NUMBER / UINT8_BIT_NUMBER;
  at::Tensor result;
  at::Tensor mask;

  if (!train) {
    result = self.clone();
    mask = at::ones({length}, self.options().dtype(at::kByte));
    return std::tie(result, mask);
  }

  result = OpPreparation::ApplyTensorWithoutFormat(self);
  mask = OpPreparation::ApplyTensorWithoutFormat({length}, self.options().dtype(at::kByte));
  at::IntArrayRef shapeArray(self.sizes());

  // DropOutGenMask use seed and seed1 to generator a seed, like this:
  //  seed1   seed
  // 127~64   63~0
  // so, we set seed1 = 0 to ensure the seed which user set is equal to the seed
  // used by the operator DropOutGenMask
  const auto gen = at_npu::detail::getDefaultNPUGenerator();
  auto pair = at::check_generator<NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
  // At present, the default value of random number may be very large,
  // which will cause overflow in graph mode, so we set seed = 0 to avoid it.
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;

  auto original_stream = c10_npu::getCurrentNPUStream();
  {
    // During the life cycle of this raii instance, the calcu stream is set as the
    // secondary stream, and tasks are distributed to the secondary stream. At the
    // same time, according to the one-stream-one-pool principle, memory is also
    // alloced from the pool of the secondary stream.
    c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
    EXEC_NPU_CMD(aclnnDropoutGenMask, shapeArray, p, seed, offset, mask);
  }
  // When tasks on multiple streams read and write the same block of memory,
  // recordStream needs to be called to ensure the correctness of memory reuse.
  c10_npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);

  EXEC_NPU_CMD(aclnnDropoutDoMask, self, mask, p, result);
  return std::tie(result, mask);
}

std::tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::native_dropout(const at::Tensor& input, double p,
                                                                           c10::optional<bool> train) {
  DO_COMPATIBILITY(aclnnDropoutGenMask, NPUNativeFunctions::native_dropout(input, p, train));
  DO_COMPATIBILITY(aclnnDropoutDoMask, NPUNativeFunctions::native_dropout(input, p, train));

  bool dropout_train = !train.has_value() ? true : train.value();
  return _dropout_npu(input, p, dropout_train);
}

at::Tensor NPUNativeOpApiFunctions::native_dropout_backward(const at::Tensor& grad_output, const at::Tensor& mask,
                                                            double scale) {
  DO_COMPATIBILITY(aclnnDropoutDoMask, NPUNativeFunctions::native_dropout_backward(grad_output, mask, scale));

  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(grad_output);
  double p = (scale == 0.0) ? 1 : (1 - 1 / scale);
  EXEC_NPU_CMD(aclnnDropoutDoMask, grad_output, mask, p, result);
  return result;
}

}  // namespace native
}  // namespace at_npu
