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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include <third_party/acl/inc/op_proto/split_combination_ops.h>

namespace at_npu {
namespace native {

namespace {
at_npu::native::DynamicInputRegFunc stack_func =
    [](DyNumAndIndex num_and_index, std::string op_name) -> ge::OperatorPtr {
      auto ge_op = std::make_shared<ge::op::Pack>(op_name.c_str());
      ge_op->create_dynamic_input_byindex_x(
          num_and_index.front().first, num_and_index.front().second);
      return ge_op;
    };
}

at::SmallVector<int64_t, SIZE> stack_npu_output_size(
    at::TensorList tensors,
    int64_t dim) {
  dim = CalcuOpUtil::make_wrap_dim(dim, tensors[0].dim() + 1);
  at::SmallVector<int64_t, SIZE> shape;
  for (int i = 0; i < dim; i++) {
    shape.emplace_back(tensors[0].size(i));
  }
  shape.emplace_back(tensors.size());
  for (int i = dim; i < tensors[0].dim(); i++) {
    shape.emplace_back(tensors[0].size(i));
  }

  return shape;
}

at::Tensor& stack_out_npu_nocheck(at::TensorList tensors, int64_t dim, at::Tensor& result) {
  auto inputTensors = CalcuOpUtil::ConvertTensorListToSmallVector(tensors);
  auto dynamic_num = inputTensors.size();

  OpCommand cmd;
  cmd.Name("Pack")
      .DynamicInputReg(stack_func, {{dynamic_num, 0}});
  for (int i = 0; i < dynamic_num; i++) {
    string inputName = "x" + std::to_string(i);
    cmd.Input(inputTensors[i],inputName);
  }
  cmd.Output(result)
    .Attr("N", (int64_t)tensors.size())
    .Attr("axis", dim)
    .Run();

  return result;
}

at::Tensor& NPUNativeFunctions::stack_out(at::TensorList tensors, int64_t dim, at::Tensor& result) {
  auto outputSize = stack_npu_output_size(tensors, dim);

  OpPreparation::CheckOut(
      {tensors[0]}, 
      result, 
      ACL_FORMAT_ND, 
      tensors[0].scalar_type(), 
      outputSize); 

  stack_out_npu_nocheck(tensors, dim, result); 

  return result;
}

at::Tensor NPUNativeFunctions::stack(at::TensorList tensors, int64_t dim) {
  auto outputSize = stack_npu_output_size(tensors, dim);

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize,
      tensors[0].options(),
      ACL_FORMAT_ND);

  stack_out_npu_nocheck(tensors, dim, result);

  return result;
}

} // namespace native
} // namespace at_npu
