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

#include <limits.h>
#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/npu/NPUGenerator.h"
#include "ATen/Utils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& random_out_npu(Tensor& result, Tensor& self, int64_t from, int64_t to, Generator* gen_) {
  auto gen = get_generator_or_default<NPUGenerator>(gen_, at::npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  SmallVector<int64_t, N> seed_list = {seed};
  SmallVector<int64_t, N> offset_list = {0, offset};
  const int32_t alg = 1;
  OpCommand cmd;
  cmd.Name("StatelessRandomUniformV2")
      .Input(self.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(seed_list, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(offset_list, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(at::Scalar(alg), at::ScalarType::Int);
  // StatelessRandomUniformV2 doesn't support int output
  if (isIntegralType(self.scalar_type(), true)) {
    Tensor resultInt = OpPreparation::ApplyTensor(self, self.options().dtype(at::kFloat));
    cmd.Attr("dtype", at::kFloat)
        .Output(resultInt)
        .Run();
    // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
    resultInt = resultInt.mul(to).sub(resultInt.mul(from).sub(static_cast<float>(from)));
    result = resultInt.to(self.scalar_type());
  } else {
    cmd.Attr("dtype", self.scalar_type())
        .Output(result)
        .Run();
    // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
    result = result.mul(to).sub(result.mul(from).sub(static_cast<float>(from)));
    // round off numbers
    result = result.to(at::kLong).to(self.scalar_type());
  }
  return result;
}

Tensor& random_npu_(Tensor& self, int64_t from, int64_t to, Generator* gen_) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    random_out_npu(contiguousSelf, contiguousSelf, from, to, gen_);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    random_out_npu(self, self, from, to, gen_);
  }
  return self;
}

Tensor& random_npu_(Tensor& self, int64_t from, c10::optional<int64_t> to, Generator* gen_) {
  int64_t to_ = to.value();
  random_npu_(self, from, to_, gen_);
  return self;
}

Tensor& random_npu_(Tensor& self, int64_t to, Generator* gen_) {
  int64_t from = 0;
  random_npu_(self, from, to, gen_);
  return self;
}

Tensor& random_npu_(Tensor& self, Generator* gen_) {
  // Check the dtype of input
  TORCH_CHECK(
      self.dtype() == at::kHalf ||
      self.dtype() == at::kFloat ||
      self.dtype() == at::kInt ||
      self.dtype() == at::kLong,
      "the dtype of input must be float16, float32, int32, int64");
  
  int64_t from = 0;
  int64_t to = 1;
  
  if (self.dtype() == at::kHalf) {
    to = NPU_HALF_MAX;
  } else if (self.dtype() == at::kInt) {
    to = INT_MAX;
  } else if (self.dtype() == at::kLong || self.dtype() == at::kFloat) {
    // the max of 'to' is also LONG_MAX because to's dtype is int64 though self is of fp32
    to = LONG_MAX;
  } 

  random_npu_(self, from, to, gen_);

  return self;
}
} // namespace native
} // namespace at
