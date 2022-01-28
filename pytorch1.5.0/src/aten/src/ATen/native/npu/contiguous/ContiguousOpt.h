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

#include <c10/npu/OptionsManager.h>
#include <torch/csrc/autograd/record_function.h>
#include <ATen/native/npu/utils/KernelNpuOutputSize.h>
#include <ATen/native/npu/contiguous/contiguous_register.h>
#include <ATen/native/npu/utils/OpPreparation.h>
namespace at {
namespace native {
namespace npu {
class TransContiguous {
 public:
  TransContiguous() {}
  virtual ~TransContiguous() {}
  static bool CheckClone(const Tensor& src, Tensor& self);
  static ContiguousTensorDesc GetTensorDescInfo(const Tensor& src, const OptimizationCases& opt_cases=optCasesDefault);
  static bool can_optimize_(ContiguousTensorDesc& tensor_desc);
  static bool CanOptimize(ContiguousTensorDesc& tensor_desc);
  static bool CanOptimize(const Tensor& tensor, const OptimizationCases& opt_cases);
  static bool contiguous_optimize_with_anyformat_(
      Tensor& self,
      const Tensor& src,
      ContiguousTensorDesc& src_desc);
  static bool ContiguousOptimizeWithAnyFormat(
      Tensor& self,
      const Tensor& src,
      const OptimizationCases& opt_cases = optCasesAnyFormat);
  static c10::optional<Tensor> ContiguousOptimizeWithAnyFormat(
      const Tensor& src,
      const OptimizationCases& opt_cases = optCasesAnyFormat);
  static bool ContiguousOptimizeWithBaseFormat(
      Tensor& self,
      const Tensor& src,
      const OptimizationCases& opt_cases = optCasesDefault,
      bool OpenCombined = true);

 private:
  static OptimizationCases optCasesDefault;
  static OptimizationCases optCasesAnyFormat;
};

} // namespace npu
} // namespace native
} // namespace at

#endif