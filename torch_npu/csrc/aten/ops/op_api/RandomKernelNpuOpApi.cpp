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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

namespace {

// RANDOM_DOUBLE_MAX = 1 << 53
const int64_t RANDOM_DOUBLE_MAX = 9007199254740992;
const int64_t RANDOM_HALF_MAX = 1 << 11;
const int64_t RANDOM_FLOAT_MAX = 1 << 24;

}  // namespace

at::Tensor& random_op_api_(at::Tensor& self, int64_t from, int64_t to, c10::optional<at::Generator> gen_) {
  auto gen = at::get_generator_or_default<NPUGeneratorImpl>(gen_, at_npu::detail::getDefaultNPUGenerator());
  auto pair = gen->philox_engine_inputs(10);
  EXEC_NPU_CMD(aclnnInplaceRandom, self, from, to, pair.first, pair.second);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::random_(at::Tensor& self, int64_t from, c10::optional<int64_t> to,
                                             c10::optional<at::Generator> gen_) {
  DO_COMPATIBILITY(aclnnInplaceRandom, NPUNativeFunctions::random_(self, from, to, gen_));
  int64_t to_ = to.value();
  random_op_api_(self, from, to_, gen_);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::random_(at::Tensor& self, int64_t to, c10::optional<at::Generator> gen_) {
  DO_COMPATIBILITY(aclnnInplaceRandom, NPUNativeFunctions::random_(self, to, gen_));
  int64_t from = 0;
  random_op_api_(self, from, to, gen_);
  return self;
}

at::Tensor& NPUNativeOpApiFunctions::random_(at::Tensor& self, c10::optional<at::Generator> gen_) {
  DO_COMPATIBILITY(aclnnInplaceRandom, NPUNativeFunctions::random_(self, gen_));
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
  random_op_api_(self, from, to, gen_);
  return self;
}
}  // namespace native
}  // namespace at_npu
