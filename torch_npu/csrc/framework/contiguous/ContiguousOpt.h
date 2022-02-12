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

#ifndef __PULGIN_NATIVE_CONTIGUOUS_CONTIGUOUS_OPTIMIZE__
#define __PULGIN_NATIVE_CONTIGUOUS_CONTIGUOUS_OPTIMIZE__

#include "torch_npu/csrc/register/OptionsManager.h"
#include <ATen/record_function.h>

#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/framework/contiguous/contiguous_register.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu
{
    namespace native
    {

        class TransContiguous
        {
        public:
            TransContiguous() {}
            virtual ~TransContiguous() {}
            static std::vector<string> FindMatchOptimizationsKeywords(
                const at::Tensor &tensor);
            static bool CheckClone(const at::Tensor &src, at::Tensor &self);
            static bool CanOptimize(const at::Tensor &src, std::vector<string> optimizations);
            static bool ContiguousOptimizeWithAnyFormat(
                at::Tensor &self,
                const at::Tensor &src,
                const std::vector<string> &optimizations = optimizations_any_format);
            static c10::optional<at::Tensor> ContiguousOptimizeWithAnyFormat(
                const at::Tensor &src,
                const std::vector<string> &optimizations = optimizations_any_format);
            static bool ContiguousOptimizeWithBaseFormat(
                at::Tensor &self,
                const at::Tensor &src,
                std::vector<string> optimizations = optimizations_default,
                bool OpenCombined = true);

        private:
            static const std::vector<string> optimizations_default;
            static const std::vector<string> optimizations_any_format;
        };

    } // namespace native
} // namespace at_npu

#endif