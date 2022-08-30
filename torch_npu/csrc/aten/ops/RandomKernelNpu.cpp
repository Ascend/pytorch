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
  constexpr int MAX_ERROR_OF_FP32_TO_FP16 = 16;
}

at::Tensor& random_out_npu(at::Tensor& result, at::Tensor& self, int64_t from, int64_t to, c10::optional<at::Generator> gen_) {
  at::Tensor SelfCopy = OpPreparation::ApplyTensor(self,self.options().dtype(at::kFloat));
  at::Tensor resultTemp1 = OpPreparation::ApplyTensor(SelfCopy); 
  at::Tensor resultTemp2 = OpPreparation::ApplyTensor(SelfCopy); 
  at::Tensor resultTemp3 = OpPreparation::ApplyTensor(SelfCopy);  

  at::IntArrayRef selfShape = self.sizes(); 
  const auto gen = at_npu::detail::getDefaultNPUGenerator();
  auto pair = at::check_generator<NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  at::SmallVector<int64_t, N> key = {seed}; 
  at::SmallVector<int64_t, N> counter = {0, offset}; 
  const int32_t alg = 1;
  OpCommand cmd1;
  cmd1.Name("StatelessRandomUniformV2")
       .Input(selfShape, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
       .InputForUint64(key, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
       .InputForUint64(counter, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
       .Input(at::Scalar(alg), at::ScalarType::Int)
       .Attr("dtype", at::kFloat)
       .Output(resultTemp1)
       .Run();
       
  OpCommand cmd2;
  cmd2.Name("Muls")
       .Input(resultTemp1)
       .Attr("value", static_cast<float>(to)-static_cast<float>(from))
       .Output(resultTemp2)
       .Run();
       
  resultTemp2.add_(static_cast<float>(from));
  resultTemp3.copy_(resultTemp2); 
  resultTemp3 = resultTemp3.to(self.dtype());
  result.copy_(resultTemp3);
  if(self.dtype() == at::kInt || self.dtype() == at::kLong) {
    OpCommand cmd3;
    cmd3.Name("Round")
        .Input(resultTemp3)
        .Output(result)
        .Run();
  }
  return result;
}

at::Tensor& random_npu_(at::Tensor& self, int64_t from, int64_t to, c10::optional<at::Generator> gen_) {
  at::Tensor selfCopy = self;
  if (self.scalar_type() == at::ScalarType::Half) {
    selfCopy = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }

  if (!NpuUtils::check_match(&selfCopy)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(selfCopy);
    at::Tensor result = random_out_npu(contiguousSelf, contiguousSelf, from, to, gen_);
    NpuUtils::format_fresh_view(selfCopy, result);
  } else {
    random_out_npu(selfCopy, selfCopy, from, to, gen_);
  }
  self.copy_(selfCopy);
  return self;
}

at::Tensor& NPUNativeFunctions::random_(at::Tensor& self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> gen_) {
  int64_t to_ = to.value();

  random_npu_(self, from, to_, gen_);

  // fp32 casting to fp16 will introduce error, so needing to counteract it.
  if (self.scalar_type() == at::ScalarType::Half) {
    self = at::where(self == to_, self - MAX_ERROR_OF_FP32_TO_FP16, self);
  }

  return self;
}

at::Tensor& NPUNativeFunctions::random_(at::Tensor& self, int64_t to, c10::optional<at::Generator> gen_) {
  int64_t from = 0;

  random_npu_(self, from, to, gen_);

  // fp32 casting to fp16 will introduce error, so needing to counteract it.
  if (self.scalar_type() == at::ScalarType::Half) {
    self = at::where(self == to, self - MAX_ERROR_OF_FP32_TO_FP16, self);
  }

  return self;
}

at::Tensor& NPUNativeFunctions::random_(at::Tensor& self, c10::optional<at::Generator> gen_) {
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
} // namespace at_npu