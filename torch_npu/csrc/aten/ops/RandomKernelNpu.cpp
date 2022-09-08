// Copyright (c) 2022 Huawei Technologies Co., Ltd
// Copyright (c) 2022, Facebook CORPORATION. 
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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace at_npu {
namespace native {

namespace {

// RANDOM_DOUBLE_MAX = 1 << 53
const int64_t RANDOM_DOUBLE_MAX = 9007199254740992;
const int64_t RANDOM_HALF_MAX = 1 << 11;
const int64_t RANDOM_FLOAT_MAX = 1 << 24;

}

at::Tensor& random_out_npu(
    at::Tensor& result,
    at::Tensor& self,
    int64_t from,
    int64_t to,
    c10::optional<at::Generator> gen_) {
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(gen_, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  at::SmallVector<int64_t, N> key = {seed}; 
  at::SmallVector<int64_t, N> counter = {0, offset}; 
  const int32_t alg = 1;
  OpCommand cmd;
  cmd.Name("StatelessRandomUniformV2")
      .Input(self.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(key, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(counter, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(at::Scalar(alg), at::ScalarType::Int);
  // StatelessRandomUniformV2 doesn't support int output
  if (isIntegralType(self.scalar_type(), true)) {
    at::Tensor resultInt = OpPreparation::ApplyTensor(self, self.options().dtype(at::kFloat));
    cmd.Attr("dtype", at::kFloat)
        .Output(resultInt)
        .Run();
    // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
    resultInt = resultInt.mul(to).sub(resultInt.mul(from).sub(static_cast<float>(from)));
    result = NPUNativeFunctions::npu_dtype_cast(resultInt, self.scalar_type());
  } else {
    cmd.Attr("dtype", self.scalar_type())
        .Output(result)
        .Run();
    // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
    result = result.mul(to).sub(result.mul(from).sub(static_cast<float>(from)));
    // round off numbers
    result = NPUNativeFunctions::npu_dtype_cast(
        NPUNativeFunctions::npu_dtype_cast(result, at::kLong), self.scalar_type());
  }
  return result;
}

at::Tensor& random_npu_(at::Tensor& self, int64_t from, int64_t to, c10::optional<at::Generator> gen_) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    random_out_npu(contiguousSelf, contiguousSelf, from, to, gen_);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    random_out_npu(self, self, from, to, gen_);
  }
  return self;
}

at::Tensor& NPUNativeFunctions::random_(
    at::Tensor& self, int64_t from,
    c10::optional<int64_t> to,
    c10::optional<at::Generator> gen_) {
  int64_t to_ = to.value();
  random_npu_(self, from, to_, gen_);
  return self;
}

at::Tensor& NPUNativeFunctions::random_(at::Tensor& self, int64_t to, c10::optional<at::Generator> gen_) {
  int64_t from = 0;
  random_npu_(self, from, to, gen_);
  return self;
}

at::Tensor& NPUNativeFunctions::random_(at::Tensor& self, c10::optional<at::Generator> gen_) {
  int64_t from = 0;
  int64_t to = 1;
  
  if (self.dtype() == at::kHalf) {
    to = RANDOM_HALF_MAX + 1;
  } else if (self.dtype() == at::kFloat) {
    to = RANDOM_FLOAT_MAX + 1;
  } else if (self.dtype() == at::kDouble) {
    to = RANDOM_DOUBLE_MAX + 1;
  } else if (self.dtype() == at::kInt) {
    to = INT_MAX;
  } else if (self.dtype() == at::kShort) {
    to = SHRT_MAX + 1;
  } else if (self.dtype() == at::kChar) {
    to = SCHAR_MAX + 1;
  } else if (self.dtype() == at::kByte) {
    to = UCHAR_MAX + 1;
  } else if (self.dtype() == at::kLong) {
    to = LONG_MAX;
  } 

  random_npu_(self, from, to, gen_);

  return self;
}
} // namespace native
} // namespace at_npu
