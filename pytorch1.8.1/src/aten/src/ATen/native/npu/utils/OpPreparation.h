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

#ifndef __NATIVE_NPU_UTILS_OP_PREPARATION__
#define __NATIVE_NPU_UTILS_OP_PREPARATION__


#include "ATen/native/npu/utils/NPUDefinition.h"
#include "ATen/native/npu/frame/OpCommandBase.h"

namespace at {
namespace native {
namespace npu {

class OpPreparation {
public:
  static UnifiedResult binary_op_check(
      Tensor& out,
      const Tensor& a,
      const Tensor& b,
      bool check_mem_overlap);
  static UnifiedResult binary_op_check(
      Tensor& out,
      const Tensor& a,
      const Scalar b,
      bool check_mem_overlap);
  static UnifiedResult comparison_op_check(
      Tensor& out,
      const Tensor& a,
      const Tensor& b,
      bool check_mem_overlap);
  static UnifiedResult unary_op_check(
      Tensor& out,
      const Tensor& a,
      bool check_mem_overlap);
  static void nullary_op(Tensor& out);
  static UnifiedResult reduce_op_check(
      Tensor& out, const Tensor& a);
  static UnifiedResult reduce_op_check(
      Tensor& out1, Tensor& out2, const Tensor& a);

  static void CheckOut(
      const std::initializer_list<Tensor>& inputs,
      Tensor& output, Tensor dst);
  static void CheckOut(
      const std::initializer_list<Tensor>& inputs,
      Tensor& output, Tensor dst,
      IntArrayRef shape);
  static void CheckOut(
      const std::initializer_list<Tensor>& input,
      Tensor& output, int64_t format,
      ScalarType dtype, IntArrayRef shape);

  static Tensor CastBackToOriFormat(const Tensor& tensor);
  static Tensor& CastBackToOriFormat(Tensor& tensor);
  // used to apply output tensor
  static Tensor ApplyTensor(const Tensor& src);
  static Tensor ApplyTensor(const Tensor& src, IntArrayRef sizes);
  static Tensor ApplyTensor(const Tensor& src, const TensorOptions& options);
  static Tensor ApplyTensor(IntArrayRef sizes, const TensorOptions& options, const Tensor& src);
  static Tensor ApplyTensorWithFormat(const Tensor& src, int64_t format);
  static Tensor ApplyTensorWithFormat(const Tensor& src, IntArrayRef sizes, int64_t format);
  static Tensor ApplyTensorWithFormat(IntArrayRef sizes, const TensorOptions& options, int64_t format);
  static Tensor ApplyTensorWithSizes(IntArrayRef sizes, const TensorOptions& options);
  // check memory
  static void CheckMemory(const std::initializer_list<Tensor>& inputs, const std::initializer_list<Tensor>& outputs);
}; // namespace OpPreparation


} // namespace npu
} // namespace native
} // namespace at

#endif