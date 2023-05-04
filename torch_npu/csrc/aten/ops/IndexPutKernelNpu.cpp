// Copyright (c) 2022 Huawei Technologies Co., Ltd
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
#include <ATen/native/TypeProperties.h>

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
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
    const at::SmallVector<int64_t, N> masks) {
  // using aicore when index is continous, otherwise aicpu
  bool is_zero_in_masks = false;
  for (int32_t i = 0; i < masks.size(); i++) {
    if (is_zero_in_masks && masks[i] == 1) {
      return true;
    }
    if (masks[i] == 0) {
      is_zero_in_masks = true;
    }
  }
  // using aicpu when indices num is more than 20000 or the type of self tensor is double.
  if (self.scalar_type() == at::kDouble || allDefinedIndices[0].numel() > 20000) {
    return true;
  }

  // first return false, after trans to int, and then check by aicpu or aicore
  for (int32_t i = 0; i < allDefinedIndices.size(); i++) {
    if (allDefinedIndices[i].scalar_type() == at::kBool) {
      return false;
    }
  }
  // indices may need broadcast, in this case, indexput is implemented by aicpu
  for (int32_t i = 1; i < allDefinedIndices.size(); i++) {
    if (allDefinedIndices[0].dim() != allDefinedIndices[i].dim()) {
      return true;
    }
    for (int32_t j = 0; j < allDefinedIndices[0].dim(); j++) {
      if (allDefinedIndices[0].sizes()[j] != allDefinedIndices[i].sizes()[j]) {
        return true;
      }
    }
  }

  int tail_size = 1;
  for (int32_t i = allDefinedIndices.size(); i < self.dim(); i++) {
    tail_size = tail_size * self.sizes()[i];
  }
  if (self.scalar_type() != at::kHalf && self.scalar_type() != at::kFloat &&
      (allDefinedIndices[0].numel() > 200 || tail_size > 128)) {
        return true;
  }
  return false;
}
at::Tensor& index_put_aicore_nocheck(
    at::Tensor& self,
    const std::vector<at::Tensor>& allDefinedIndices,
    at::SmallVector<int64_t, N> masks,
    at::SmallVector<int64_t, N> expand_masks,
    const at::Tensor& value,
    bool accumulate) {
  if (value.numel() == 0) {
    return self;
  }
  at::Tensor tempSelf = self;
  at::Tensor tempValue = value;
  if (self.scalar_type() == at::ScalarType::Half) {
    tempSelf = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    tempValue = NPUNativeFunctions::npu_dtype_cast(value, at::ScalarType::Float);
  }
  at::Tensor tempValue_broadcast = tempValue;
  if (self.dim() == 1 && allDefinedIndices.size() == 1 && allDefinedIndices[0].scalar_type() == at::kLong &&
      allDefinedIndices[0].sizes()[0] != value.sizes()[0]) {
        tempValue_broadcast = NPUNativeFunctions::npu_broadcast(tempValue, allDefinedIndices[0].sizes());
  }
  auto masks_tensors = at::tensor(masks, self.options().dtype(at::kLong));
  auto expand_masks_tensors = at::tensor(expand_masks, self.options().dtype(at::kLong));
  OpCommand cmd;
  cmd.Name("IndexPutV2")
      .Input(tempSelf, (string)"x")
      .Input(tempValue_broadcast, (string)"value")
      .Input(masks_tensors, (string)"indexed_sizes")
      .Input(expand_masks_tensors, (string)"indexed_strides");
  for (int i = 0; i < allDefinedIndices.size(); i++) {
    string inputName = "indices" +std::to_string(i);
    cmd.Input(allDefinedIndices[i], inputName);
  }
  cmd.DynamicInputReg(indexput_func<ge::op::IndexPutV2>, {{allDefinedIndices.size(), 4}})
      .Output(tempSelf, (string)"x")
      .Attr("accumulate", accumulate)
      .Run();
  if (self.scalar_type() == at::ScalarType::Half) {
    tempSelf = NPUNativeFunctions::npu_dtype_cast(tempSelf, at::ScalarType::Half);
    self.copy_(tempSelf);
  } else {
    self = tempSelf;
  }
  return self;
}

at::SmallVector<int64_t, N> npu_expand_tensors_mask(
    const at::Tensor& self,
    const torch::List<c10::optional<at::Tensor>>& indices) {
  at::SmallVector<int64_t, N> result;
  for (c10::optional<at::Tensor> index_opt : indices) {
    if (!index_opt.has_value()) {
      result.emplace_back(0);
    } else {
      const auto& index = *index_opt;
      if (index.scalar_type() != at::kByte && index.scalar_type() != at::kBool) {
        result.emplace_back(0);
        break;
      }
    }
  }
  if (result.empty()) {
    result.emplace_back(1);
  }
  return result;
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
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    bool accumulate) {
  return self.clone(at::MemoryFormat::Contiguous)
      .index_put_(indices, value, accumulate);
}
at::Tensor& index_put_aicpu(
    at::Tensor& result,
    at::Tensor& self,
    std::vector<at::Tensor> allDefinedIndices,
    at::SmallVector<int64_t, N> masks,
    const at::Tensor& value,
    bool accumulate) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    result = index_put_aicpu_nocheck(
        contiguousSelf, contiguousSelf, allDefinedIndices, masks, value, accumulate);
  } else {
    result = index_put_aicpu_nocheck(self, self, allDefinedIndices, masks, value, accumulate);
  }
  return result;
}

at::Tensor& index_put_aicore(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    std::vector<at::Tensor> allDefinedIndices,
    const at::Tensor& value,
    bool accumulate) {
  std::vector<at::Tensor> indices_expand;
  at::SmallVector<int64_t, N> indices_expand_mask;
  at::Tensor value_broadcast;
  auto input_shape = array_to_small_vector(self.sizes());
  c10::List<c10::optional<at::Tensor>> shaped_indices;

  // transfer input and index to 1 dim tensor if index is bool type, and the shape of input and index is the same
  if (allDefinedIndices.size() == 1) {
    auto indices_shape = array_to_small_vector(allDefinedIndices[0].sizes());
    if (allDefinedIndices[0].scalar_type() == at::kBool && input_shape == indices_shape) {
      self = self.reshape(-1);
      shaped_indices.push_back(allDefinedIndices[0].reshape(-1));
      indices_expand = AdvanceIndex::npu_expand_tensors(self, shaped_indices);
      indices_expand_mask = npu_expand_tensors_mask(self, shaped_indices);
    }
  }
  if (shaped_indices.size() == 0) {
    indices_expand = AdvanceIndex::npu_expand_tensors(self, indices);
    indices_expand_mask = npu_expand_tensors_mask(self, indices);
  }

  // value broadcast
  auto index_output_size = index_npu_output_size(self, indices_expand);
  auto value_shape = array_to_small_vector(value.sizes());
  value_broadcast = (index_output_size != value_shape) ?
      NPUNativeFunctions::npu_broadcast(value, index_output_size) : value;

  // re-caculate mask
  at::SmallVector<int64_t, N> masks;
  for (c10::optional<at::Tensor> index_opt : indices_expand) {
    if (index_opt.has_value()) {
      const auto& index = *index_opt;
      if (index.defined()) {
        masks.emplace_back(1);
      } else {
        masks.emplace_back(0);
      }
    } else {
      masks.emplace_back(0);
    }
  }

  // after expand, check it through aicpu or aicore
  bool aicpu_true = is_aicpu_valid(self, indices_expand, masks);
  if (aicpu_true) {
    index_put_aicpu(self, self, indices_expand, masks, value, accumulate);
    self.copy_(self);
  } else {
    if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
      index_put_aicore_nocheck(contiguousSelf, indices_expand, masks, indices_expand_mask, value_broadcast, accumulate);
      self.copy_(contiguousSelf);
    } else {
      index_put_aicore_nocheck(self, indices_expand, masks, indices_expand_mask, value_broadcast, accumulate);
    }
  }
  if (shaped_indices.size() != 0) {
    self = self.reshape(input_shape);
  }
  return self;
}

at::Tensor& NPUNativeFunctions::index_put_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate) {
  return at::_index_put_impl_(
      self, indices, value, accumulate, false);
}

at::Tensor& NPUNativeFunctions::_index_put_impl_(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  at::native::checkIndexTensorTypes(indices);
  at::SmallVector<int64_t, N> masks;
  std::vector<at::Tensor> allDefinedIndices;
  for (c10::optional<at::Tensor> index_opt : indices) {
    if (index_opt.has_value()) {
      const auto& index = *index_opt;
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

  bool aicpu_true = is_aicpu_valid(self, allDefinedIndices, masks);
  if (aicpu_true) {
    index_put_aicpu(selfCopy, selfCopy, allDefinedIndices, masks, valueCopy, accumulate);
    self.copy_(selfCopy);
  } else {
    index_put_aicore(selfCopy, indices, allDefinedIndices, valueCopy, accumulate);
    self.copy_(selfCopy);
  }
  return self;
}

} // namespace native
} // namespace at_npu
