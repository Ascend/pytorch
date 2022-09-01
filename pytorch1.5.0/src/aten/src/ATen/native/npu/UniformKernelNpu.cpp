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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/npu/NPUGenerator.h"
#include "ATen/Utils.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& uniform_out_npu(
    Tensor& result,
    const Tensor& self,
    double from,
    double to,
    Generator* gen_) {
  auto gen = get_generator_or_default<NPUGenerator>(gen_, at::npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  const int64_t seed = pair.first;
  const int64_t offset = pair.second;
  SmallVector<int64_t, N> seed_list = {seed};
  SmallVector<int64_t, N> offset_list = {0, offset};
  int64_t alg = 1;
  OpCommand cmd;
  cmd.Name("StatelessRandomUniformV2")
      .Input(self.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Input(seed_list, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(offset_list, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT, (string)"uint64")
      .Input(at::Scalar(alg), at::ScalarType::Int)
      .Output(result)
      .Attr("dtype", self.scalar_type())
      .Run();
  // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
  result = result.mul(to).sub(result.mul(from).sub(from));
  return result;
}

Tensor& uniform_npu_(Tensor& self, double from, double to, Generator* gen_) {
  if (!NpuUtils::check_match(&self)) {
    Tensor selfContiguous = NpuUtils::format_contiguous(self);
    uniform_out_npu(selfContiguous, selfContiguous, from, to, gen_);
    NpuUtils::format_fresh_view(self, selfContiguous);
  } else {
    uniform_out_npu(self, self, from, to, gen_);
  }
  return self;
}
} // namespace native
} // namespace at
