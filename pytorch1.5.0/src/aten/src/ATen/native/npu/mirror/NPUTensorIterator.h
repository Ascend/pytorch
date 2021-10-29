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

#ifndef __NATIVE_NPU_UTILS_NPU_TENSOR_ITERATOR__
#define __NATIVE_NPU_UTILS_NPU_TENSOR_ITERATOR__

#include <ATen/ATen.h>
#include <functional>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <bitset>
#include <c10/util/TypeCast.h>

namespace at {
namespace native {
namespace npu {

struct NPUOperandInfo {
  using StrideVector = SmallVector<int64_t, 6>;
  NPUOperandInfo() {}
  explicit NPUOperandInfo(const Tensor& t) : tensor(t) {
    if (t.defined()) {
      target_dtype = t.scalar_type();
      current_dtype = target_dtype;
    }
    validate();
  }
  NPUOperandInfo(const Tensor& t, ScalarType dtype)
    : tensor(t), target_dtype(dtype), current_dtype(t.scalar_type()) {
    validate();
  }

  bool is_type_defined() const {
    return target_dtype != ScalarType::Undefined;
  }
 
  void validate() {
    TORCH_CHECK(
        !tensor.defined() || tensor.layout() == kStrided,
        "unsupported tensor layout: ", tensor.layout());
  }

  StrideVector stride_bytes;
  Tensor tensor;
  ScalarType target_dtype = ScalarType::Undefined;
  ScalarType current_dtype = ScalarType::Undefined;
  bool is_output = false;
}; // class NPUOperandInfo

enum class CommonDTypeStrategy : uint8_t {
  NONE, // Do not compute a common dtype
  CHECK, // Compute and validate a common dtype but don't promote.
  PROMOTE_INPUTS, // Promote common dtype but only validate inputs (comparison ops have boolean output)
  PROMOTE // Promote to common dtype.
};

class NPUTensorIterator {
 public:
  NPUTensorIterator() {}
  ~NPUTensorIterator() {}

  static std::tuple<ScalarType, IntArrayRef> binary_op(
      Tensor& out, 
      const Tensor& a,
      const Tensor& b,
      bool check_mem_overlap = false);
  static std::tuple<ScalarType, IntArrayRef> binary_op(
      const Tensor& a,
      const Scalar b);
  static std::tuple<ScalarType, IntArrayRef> comparison_op(
      Tensor& out, 
      const Tensor& a, 
      const Tensor& b,
      bool check_mem_overlap = false);
  static std::tuple<ScalarType, IntArrayRef> unary_op(
      Tensor& out, 
      const Tensor& a,
      bool check_mem_overlap = false);
  static void nullary_op(Tensor& out);
  static std::tuple<ScalarType, IntArrayRef> reduce_op(
      Tensor& out, 
      const Tensor& a);
  static std::tuple<ScalarType, IntArrayRef> reduce_op(
      Tensor& out1, 
      Tensor& out2, 
      const Tensor& a);
  
  int noutputs() const {
    return num_outputs_; 
  }

  IntArrayRef strides(int arg) const {
    return operands_[arg].stride_bytes;
  }
  ScalarType dtype(int arg=0) const {
    return operands_[arg].current_dtype;
  }
  ScalarType common_dtype() const { 
    return common_dtype_;
  }

  const SmallVector<NPUOperandInfo, 4> GetOperandInfo() const {
    return operands_;
  }

  /// Construction
  void add_output(const Tensor& output) {
    operands_.emplace_back(output);
    num_outputs_++;
  }

  void add_input(const Tensor& input) {
    operands_.emplace_back(input);
  }

  void promote_common_dtype() {
    common_dtype_strategy_ = CommonDTypeStrategy::PROMOTE;
  }

  void compute_common_dtype_only_for_inputs() {
    common_dtype_strategy_ = CommonDTypeStrategy::PROMOTE_INPUTS;
  }

  void compute_types();
  std::tuple<ScalarType, bool> compute_common_type();

 private:
  SmallVector<NPUOperandInfo, 4> operands_;
  int num_outputs_ = 0;
  bool promote_npu_output_dtypes_ = false;
  bool all_ops_same_shape_ = false;
  ScalarType common_dtype_ = ScalarType::Undefined;
  bool is_reduction_ = false;
  CommonDTypeStrategy common_dtype_strategy_ = CommonDTypeStrategy::CHECK;
}; // class NPUTensorIterator

} // namespace npu
} // namespace native
} // namespace at

#endif
