// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/NpuUtils.h"
#include "ATen/native/npu/utils/OpTemplate.h"
#include "c10/npu/SecondaryStreamGuard.h"
#include "c10/npu/NPUCachingAllocator.h"
#include "ATen/npu/NPUGenerator.h"

namespace at {
namespace native {
using namespace at::native::npu;
Tensor dropout_do_mask_with_byte_mask(
    Tensor& result,
    const Tensor& self,
    const Tensor& mask,
    Scalar prob) {
  OpCommand cmd;
  cmd.Name("DropOutDoMaskV3")
      .Input(self)
      .Input(mask)
      .Input(prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(result)
      .Run();
  return result;
}

void dropout_gen_byte_mask(const Tensor& self, Scalar prob, Tensor &mask) {
  IntArrayRef selfShape = self.sizes();
  OpCommand cmd;
  // If either seed or seed2 are set to be non-zero, the random number generator
  // is seeded by the given seed. Otherwise, it is seeded by a random seed.
  // DropOutGenMaskV3 use seed and seed2 to generator a seed, like this:
  //  seed2   seed
  // 127~64   63~0
  // so, we set seed2 = 0 to ensure the seed which user set is equal to the seed 
  // used by the operator DropOutGenMaskV3
  const auto gen = at::npu::detail::getDefaultNPUGenerator();
  AT_ASSERT(gen != nullptr);
  const int64_t seed = static_cast<int64_t>(gen->current_seed());
  const int64_t seed2 = 0;
  cmd.Name("DropOutGenMaskV3")
      .Input(selfShape)
      .Input(prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
      .Output(mask)
      .Attr("seed", seed)
      .Attr("seed2", seed2)
      .Run();
}

Tensor& dropout_npu_impl(
    Tensor &result,
    const Tensor& self,
    double p,
    Tensor &mask) {
  Tensor selfCp = NpuUtils::format_contiguous(self);
  TORCH_CHECK(
      p >= 0 && p <= 1,
      "dropout probability has to be between 0 and 1, but got ", p);
  TORCH_CHECK(
      at::isFloatingType(selfCp.scalar_type()),
      "dropout only supports floating-point dtypes");

  double retain = 1. - p;
  Scalar prob = Scalar(retain);
  auto original_stream = c10::npu::getCurrentNPUStream();
  {
    // During the life cycle of this raii instance, the calcu stream is set as the
    // secondary stream, and tasks are distributed to the secondary stream. At the
    // same time, according to the one-stream-one-pool principle, memory is also
    // alloced from the pool of the secondary stream.
    c10::npu::SecondaryStreamGuard guard(c10::npu::getCurrentSecondaryStream());
    dropout_gen_byte_mask(selfCp, prob, mask);
  }
  // When tasks on multiple streams read and write the same block of memory,
  // recordStream needs to be called to ensure the correctness of memory reuse.
  c10::npu::NPUCachingAllocator::recordStream(mask.storage().data_ptr(), original_stream);
  dropout_do_mask_with_byte_mask(result, selfCp, mask, prob);

  return result;
}

Tensor _dropout_with_byte_mask_npu(
    const Tensor& self,
    double p,
    Tensor &mask) {
  Tensor result = OpPreparation::ApplyTensor(self);
  dropout_npu_impl(result, self, p, mask);
  return result;
}

Tensor& _dropout_with_byte_mask_npu_(
    Tensor& self,
    double p,
    Tensor &mask) {
   dropout_npu_impl(self, self, p, mask);
   return self;
}

Tensor dropout_with_byte_mask(const Tensor& self, double p, bool train) {
  TORCH_CHECK(
      self.is_npu(),
      "dropout_with_byte_mask only supports device for NPU!");
  if (p == 0 || !train || self.numel() == 0) {
    return self;
  }
  if (p == 1) {
    return self.mul(at::zeros(self.sizes(), self.options()));
  }
  Tensor mask = at::empty_with_format(
      self.sizes(),
      self.options().dtype(at::kByte),
      ACL_FORMAT_ND);
  return at::_dropout_with_byte_mask(self, p, mask);
}

Tensor& dropout_with_byte_mask_(Tensor& self, double p, bool train) {
  TORCH_CHECK(
      self.is_npu(),
      "dropout_with_byte_mask only supports device for NPU!");
  if (p == 0 || !train || self.numel() == 0) {
    return self;
  }
  if (p == 1) {
    return self.mul_(at::zeros(self.sizes(), self.options()));
  }
  Tensor mask = at::empty_with_format(
      self.sizes(),
      self.options().dtype(at::kByte),
      ACL_FORMAT_ND);
  if (!NpuUtils::check_match(&self)) {
    Tensor result = NpuUtils::format_contiguous(self);
    at::_dropout_with_byte_mask_(result, p, mask);
    NpuUtils::format_fresh_view(self, result);
  } else {
    at::_dropout_with_byte_mask_(self, p, mask);
  }
  return self;
}

} // namespace native
} // namespace at