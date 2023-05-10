// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at_npu
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __PULGIN_NATIVE_NPU_UTILS_OP_PREPARATION__
#define __PULGIN_NATIVE_NPU_UTILS_OP_PREPARATION__

#include "torch_npu/csrc/framework/utils/NPUDefinition.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace at_npu
{
    namespace native
    {

        class OpPreparation
        {
        public:
            static UnifiedResult binary_op_check(
                at::Tensor &out,
                const at::Tensor &a,
                const at::Tensor &b,
                bool check_mem_overlap);
            static UnifiedResult binary_op_check(
                at::Tensor &out,
                const at::Tensor &a,
                const c10::Scalar b,
                bool check_mem_overlap);
            static UnifiedResult comparison_op_check(
                at::Tensor &out,
                const at::Tensor &a,
                const at::Tensor &b,
                bool check_mem_overlap);
            static UnifiedResult unary_op_check(
                at::Tensor &out,
                const at::Tensor &a,
                bool check_mem_overlap);
            static void nullary_op(at::Tensor &out);
            static UnifiedResult reduce_op_check(
                at::Tensor &out, const at::Tensor &a);
            static UnifiedResult reduce_op_check(
                at::Tensor &out1, at::Tensor &out2, const at::Tensor &a);

            static void CheckOut(
                const std::initializer_list<at::Tensor> &inputs,
                at::Tensor &output, at::Tensor dst);
            static void CheckOut(
                const std::initializer_list<at::Tensor> &inputs,
                at::Tensor &output, at::Tensor dst,
                c10::IntArrayRef shape);
            static void CheckOut(
                const std::initializer_list<at::Tensor> &input,
                at::Tensor &output, int64_t format,
                at::ScalarType dtype, c10::IntArrayRef shape);

            static at::Tensor CastBackToOriFormat(const at::Tensor &tensor);
            static at::Tensor &CastBackToOriFormat(at::Tensor &tensor);
            // used to apply output tensor
            static at::Tensor ApplyTensor(const at::Tensor &src);
            static at::Tensor ApplyTensor(const at::Tensor &src, c10::IntArrayRef sizes);
            static at::Tensor ApplyTensor(const at::Tensor &src, const c10::TensorOptions &options);
            static at::Tensor ApplyTensor(c10::IntArrayRef sizes, const c10::TensorOptions &options,
                                          const at::Tensor &src);
            static at::Tensor ApplyTensorWithFormat(const at::Tensor &src, int64_t format,
                                                    bool keep_format = false);
            static at::Tensor ApplyTensorWithFormat(const at::Tensor &src, c10::IntArrayRef sizes, int64_t format,
                                                    bool keep_format = false);
            static at::Tensor ApplyTensorWithFormat(c10::IntArrayRef sizes, const c10::TensorOptions &options,
                                                    int64_t format, bool keep_format = false);
            static at::Tensor ApplyTensorWithSizes(c10::IntArrayRef sizes, const c10::TensorOptions &options);
            // check memory
            static void CheckMemory(const std::initializer_list<at::Tensor> &inputs,
                                    const std::initializer_list<at::Tensor> &outputs);
        }; // namespace OpPreparation

    } // namespace native
} // namespace at_npu

#endif