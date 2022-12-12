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

#include <ATen/native/IndexingUtils.h>

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include <ATen/native/TypeProperties.h>
#include "torch_npu/csrc/framework/utils/AdvancedIndex.h"
#include "torch_npu/csrc/framework/graph/util/GraphModeGuard.h"
#include "torch_npu/csrc/framework/graph/construct/GraphConstructor.h"
#include <third_party/acl/inc/op_proto/experiment_ops.h>

namespace at_npu {
namespace native {

namespace 
{
  template <typename ge_op_type>
  at_npu::native::DynamicInputRegFunc indexput_func =
      [](DyNumAndIndex num_and_index,
        std::string op_name) -> ge::OperatorPtr 
        {
          auto ge_op = std::make_shared<ge_op_type>(op_name.c_str());
          ge_op->create_dynamic_input_byindex_indices(
              num_and_index.front().first, num_and_index.front().second);
          return ge_op;
        };
}
bool is_aicpu_valid(const at::Tensor& self,
    const std::vector<at::Tensor>& allDefinedIndices,
    const at::SmallVector<int64_t, N> masks,
    const at::Tensor& value) {
  bool flag = true;
  // allDefinedIndices size is more than two or the type of self tensor is double, implemented by AICPU
  if (allDefinedIndices.size() > 2 || self.scalar_type() == at::kDouble) {
    return true;
  }
  // allDefinedIndices has only one tensor and is a bool type, implemented by AICore.
  if (allDefinedIndices.size() == 1 && allDefinedIndices[0].scalar_type() == at::kBool) {
    // value may need broadcast, implemented by AICPU
    if (value.sizes()[0] == 1 || value.dim() != self.dim()) {
      return true;
    }
    for (int32_t i = self.dim() - 1; i >= allDefinedIndices.size(); i--) {
      if (value.sizes()[i] != self.sizes()[i]) {
        return true;
      }
    }
    return false;
  } else {
    for (int32_t i = 0; i < allDefinedIndices.size(); i++) {
      // if all Indices tensor is int64, they are implemented by AICore; otherwise AICPU.
      if (allDefinedIndices[i].scalar_type() == at::kBool) {
        return true;
      }
    }
  }
  int32_t pre = 0;
  for (int32_t i = 0; i < masks.size(); i++) {
    // Indices tensors are at the discontinuous axis position, implemented by AICPU, otherwise AICORE
    if ((masks[i] == 0 && i != pre) || (masks[i] == 0 && i==0)) {
      return true;
    }
    if (masks[i] == 1) {
      pre = pre + 1;
    }
  }
  // value need broadcast, implemented by AICPU
  if (allDefinedIndices.size() < self.dim()) {
    if (value.dim() != self.dim()) {
      return true;
    }
    for (int32_t i = self.dim() - 1; i >= allDefinedIndices.size(); i--) {
      if (value.sizes()[i] != self.sizes()[i]){
        return true;
      }
    }
  }
  if (allDefinedIndices[0].sizes()[0] != value.sizes()[0]) {
    return true;
  }
  return false;
}
at::Tensor& index_put_aicore_nocheck(
    at::Tensor& self,
    const std::vector<at::Tensor>& allDefinedIndices,
    at::SmallVector<int64_t, N> masks,
    const at::Tensor& value,
    bool accumulate) {
  if (value.numel() == 0) {
    return self;
  }
  at::Tensor tempSelf = self;
  at::Tensor tempValue = value;
  auto masks_tensors = at::tensor(masks, self.options().dtype(at::kLong));
  OpCommand cmd;
  cmd.Name("IndexPutV2")
      .Input(tempSelf, (string)"x")
      .Input(tempValue, (string)"value")
      .Input(masks_tensors, (string)"indexed_sizes")
      .Input(masks_tensors, (string)"indexed_strides");
  for (int i = 0; i < allDefinedIndices.size(); i++) {
    string inputName = "indices" +std::to_string(i);
    cmd.Input(allDefinedIndices[i], inputName);
  }
  cmd.DynamicInputReg(indexput_func<ge::op::IndexPutV2>, {{allDefinedIndices.size(), 4}})
      .Output(tempSelf, (string)"x")
      .Attr("accumulate", accumulate)
      .Run();
  return tempSelf;
}

at::Tensor& index_put_aicpu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    std::vector<at::Tensor> allDefinedIndices,
    at::SmallVector<int64_t, N> masks,
    const at::Tensor& value,
    bool accumulate) {
  if (value.numel() == 0) {
    return result;
  }

  at::Tensor tempSelf = self;
  at::Tensor tempValue = value;
  if (self.scalar_type() == at::ScalarType::Half) {
    tempSelf = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    tempValue = NPUNativeFunctions::npu_dtype_cast(value, at::ScalarType::Float);
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Float);
  }

  OpCommand cmd;
  cmd.Name("IndexPut")
      .Input(tempSelf)
      .Input(tempValue)
      .Input(masks, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
      .Inputs(allDefinedIndices)
      .Output(result)
      .Attr("accumulate", accumulate)
      .Run();

  if (self.scalar_type() == at::ScalarType::Half) {
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Half);
  }
  return result;
}

at::Tensor NPUNativeFunctions::index_put(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>> & indices,
    const at::Tensor& value,
    bool accumulate) {
  return self.clone(at::MemoryFormat::Contiguous)
      .index_put_(indices, value, accumulate);
}

at::Tensor& NPUNativeFunctions::index_put_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>> & indices,
    const at::Tensor& value,
    const bool accumulate) {
  return at::_index_put_impl_(
      self, indices, value, accumulate, false);
}

at::Tensor& NPUNativeFunctions::_index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>> & indices,
    const at::Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  at::native::checkIndexTensorTypes(indices);
  at::SmallVector<int64_t, N> masks;
  std::vector<at::Tensor> allDefinedIndices;
  for (c10::optional<at::Tensor> index_opt : indices) {
    if (index_opt.has_value()) {
      at::Tensor index = std::move(*index_opt);
      if (index.defined()) {
        allDefinedIndices.emplace_back(index);
        masks.emplace_back(1);
      } else {
        masks.emplace_back(0);
      }
    } else {
      masks.emplace_back(0);
    }
  }
  
  OpPreparation::CastBackToOriFormat(self);
  at::Tensor valueCopy = value;
  at::Tensor selfCopy = self;
  OpPreparation::CastBackToOriFormat(valueCopy);
  bool aicpu_true = is_aicpu_valid(self, allDefinedIndices, masks, value);
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(selfCopy);
    if (aicpu_true) {
      at::Tensor result = index_put_aicpu_nocheck(
          contiguousSelf, contiguousSelf, allDefinedIndices, masks, valueCopy, accumulate);
      self.copy_(result);
    } else {
      index_put_aicore_nocheck(contiguousSelf, allDefinedIndices, masks, valueCopy, accumulate);
      self.copy_(contiguousSelf);
    }
    
  } else {
    if (aicpu_true) {
      index_put_aicpu_nocheck(selfCopy, selfCopy, allDefinedIndices, masks, valueCopy, accumulate);
      self.copy_(selfCopy);
    } else {
      index_put_aicore_nocheck(selfCopy, allDefinedIndices, masks, valueCopy, accumulate);
      self.copy_(selfCopy);
    } 
  }
  return self;
}
} // namespace native
} // namespace at_npu
