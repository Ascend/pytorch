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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <limits.h>
#include <ATen/NamedTensorUtils.h>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::bernoulli_(at::Tensor& self, double p, c10::optional<at::Generator> gen) {
  DO_COMPATIBILITY(aclnnInplaceBernoulli, NPUNativeFunctions::bernoulli_(self, p, gen));
  auto gen_ = at::get_generator_or_default<NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;

  const c10::Scalar& pScalar = at::Scalar(p);
  EXEC_NPU_CMD(aclnnInplaceBernoulli, self, pScalar, seed, offset);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::bernoulli_(at::Tensor& self, const at::Tensor& p,
                                                c10::optional<at::Generator> gen) {
  DO_COMPATIBILITY(aclnnInplaceBernoulliTensor, NPUNativeFunctions::bernoulli_(self, p, gen));
  auto gen_ = at::get_generator_or_default<NPUGeneratorImpl>(gen, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen_->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;

  EXEC_NPU_CMD(aclnnInplaceBernoulliTensor, self, p, seed, offset);
  return self;
}

at::Tensor NPUNativeOpApiFunctions::bernoulli(const at::Tensor& self, c10::optional<at::Generator> gen) {
  DO_COMPATIBILITY(aclnnInplaceBernoulliTensor, NPUNativeFunctions::bernoulli(self, gen));
  at::Tensor self_copy = OpPreparation::ApplyTensorWithoutFormat(self);
  return NPUNativeOpApiFunctions::bernoulli_(self_copy, self, gen);
}

at::Tensor NPUNativeOpApiFunctions::bernoulli(const at::Tensor& self, double p, c10::optional<at::Generator> gen) {
  DO_COMPATIBILITY(aclnnInplaceBernoulli, NPUNativeFunctions::bernoulli(self, p, gen));
  return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT).bernoulli_(p, gen);
}

at::Tensor& NPUNativeOpApiFunctions::bernoulli_out(const at::Tensor& self, c10::optional<at::Generator> gen,
                                                   at::Tensor& result) {
  DO_COMPATIBILITY(aclnnInplaceBernoulliTensor, NPUNativeFunctions::bernoulli_out(self, gen, result));
  result.resize_(self.sizes()).bernoulli_(self, gen);
  at::namedinference::propagate_names(result, self);
  return result;
}
}  // namespace native
}  // namespace at_npu
