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

#ifndef __NATIVE_NPU_CONTIGUOUS_CONTIGUOUS_OPTIMIZE__
#define __NATIVE_NPU_CONTIGUOUS_CONTIGUOUS_OPTIMIZE__

#include <ATen/native/npu/utils/NpuUtils.h>
#include <c10/npu/OptionsManager.h>
#include <ATen/record_function.h>
#include "ATen/native/npu/contiguous/contiguous_register.h"

namespace at {
namespace native {
namespace npu {

class TransContiguous {
 public:
  TransContiguous() {}
  virtual ~TransContiguous() {}
  static std::vector<string> FindMatchOptimizationsKeywords(
      const Tensor& tensor);
  static bool CheckClone(const Tensor& src, Tensor& self);
  static bool CanOptimize(const Tensor& src, std::vector<string> optimizations);
  static bool ContiguousOptimizeWithAnyFormat(
      Tensor& self,
      const Tensor& src,
      std::vector<string> optimizations = optimizations_any_format);
  static c10::optional<Tensor> ContiguousOptimizeWithAnyFormat(
      const Tensor& src,
      std::vector<string> optimizations = optimizations_any_format);
  static bool ContiguousOptimizeWithBaseFormat(
      Tensor& self,
      const Tensor& src,
      std::vector<string> optimizations = optimizations_default,
      bool OpenCombined = true);

 private:
  static const std::vector<string> optimizations_default;
  static const std::vector<string> optimizations_any_format;
};

} // namespace npu
} // namespace native
} // namespace at

#endif