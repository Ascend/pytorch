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

void check_size(const at::Tensor& value, const at::Tensor& self, const at::TensorList& indices) {
  auto value_shape = array_to_small_vector(value.sizes());
  auto index_output_size = index_npu_output_size(self, indices);
  size_t dims_a = value_shape.size();
  size_t dims_b = index_output_size.size();
  size_t ndim = dims_a > dims_b ? dims_a : dims_b;

  // Use ptrdiff_t to ensure signed comparison.
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    ptrdiff_t offset = ndim - 1 - i;
    ptrdiff_t dim_a = dims_a - 1 - offset;
    ptrdiff_t dim_b = dims_b - 1 - offset;
    auto size_a = (dim_a >= 0) ? value_shape[dim_a] : 1;
    auto size_b = (dim_b >= 0) ? index_output_size[dim_b] : 1;

    TORCH_CHECK(size_a == size_b || size_a == 1 || size_b == 1,
        "shape mismatch: value tensor of shape ", value.sizes(),
        " cannot be broadcast to indexing result of shape ", self.sizes());
  }
}

bool is_aicpu_valid(const at::Tensor& self,
    const std::vector<at::Tensor>& all_defined_indices,
    const at::SmallVector<int64_t, N> masks) { 
  // using aicpu at non-binary scene
  if (!env::CheckJitDisable()) {
    return true;
  }
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
  if (self.scalar_type() == at::kDouble || all_defined_indices[0].numel() > 20000) {
    return true;
  }

  // first return false, after trans to int, and then check by aicpu or aicore
  for (int32_t i = 0; i < all_defined_indices.size(); i++) {
    if (all_defined_indices[i].scalar_type() == at::kBool) {
      return false;
    }
  }
  // indices may need broadcast, in this case, indexput is implemented by aicpu
  for (int32_t i = 1; i < all_defined_indices.size(); i++) {
    if (all_defined_indices[0].dim() != all_defined_indices[i].dim()) {
      return true;
    }
    for (int32_t j = 0; j < all_defined_indices[0].dim(); j++) {
      if (all_defined_indices[0].sizes()[j] != all_defined_indices[i].sizes()[j]) {
        return true;
      }
    }
  }

  int tail_size = 1;
  for (int32_t i = all_defined_indices.size(); i < self.dim(); i++) {
    tail_size = tail_size * self.sizes()[i];
  }
  if (self.scalar_type() != at::kHalf && self.scalar_type() != at::kFloat &&
      (all_defined_indices[0].numel() > 200 || tail_size > 128)) {
        return true;
  }
  return false;
}
at::Tensor& index_put_aicore_nocheck(
    at::Tensor& self,
    const std::vector<at::Tensor>& all_defined_indices,
    at::SmallVector<int64_t, N> masks,
    at::SmallVector<int64_t, N> expand_masks,
    const at::Tensor& value,
    bool accumulate) {
  if (value.numel() == 0) {
    return self;
  }
  at::Tensor temp_self = self;
  at::Tensor temp_value = value;
  if (self.scalar_type() == at::ScalarType::Half) {
    temp_self = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    temp_value = NPUNativeFunctions::npu_dtype_cast(value, at::ScalarType::Float);
  }
  at::Tensor temp_value_broadcast = temp_value;
  if (self.dim() == 1 && all_defined_indices.size() == 1 && all_defined_indices[0].scalar_type() == at::kLong &&
      all_defined_indices[0].sizes()[0] != value.sizes()[0]) {
        temp_value_broadcast = NPUNativeFunctions::npu_broadcast(temp_value, all_defined_indices[0].sizes());
  }
  auto masks_tensors = at::tensor(masks, self.options().dtype(at::kLong));
  auto expand_masks_tensors = at::tensor(expand_masks, self.options().dtype(at::kLong));
  OpCommand cmd;
  cmd.Name("IndexPutV2")
      .Input(temp_self, (string)"x")
      .Input(temp_value_broadcast, (string)"value")
      .Input(masks_tensors, (string)"indexed_sizes")
      .Input(expand_masks_tensors, (string)"indexed_strides");
  for (int i = 0; i < all_defined_indices.size(); i++) {
    string input_name = "indices" +std::to_string(i);
    cmd.Input(all_defined_indices[i], input_name);
  }
  cmd.DynamicInputReg(indexput_func<ge::op::IndexPutV2>, {{all_defined_indices.size(), 4}})
      .Output(temp_self, (string)"x")
      .Attr("accumulate", accumulate)
      .Run();
  if (self.scalar_type() == at::ScalarType::Half) {
    temp_self = NPUNativeFunctions::npu_dtype_cast(temp_self, at::ScalarType::Half);
    self.copy_(temp_self);
  } else {
    self = temp_self;
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
    std::vector<at::Tensor> all_defined_indices,
    at::SmallVector<int64_t, N> masks,
    const at::Tensor& value,
    bool accumulate) {
  if (value.numel() == 0) {
    return result;
  }

  at::Tensor temp_self = self;
  at::Tensor temp_value = value;
  if (self.scalar_type() == at::ScalarType::Half) {
    temp_self = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    temp_value = NPUNativeFunctions::npu_dtype_cast(value, at::ScalarType::Float);
    result = NPUNativeFunctions::npu_dtype_cast(result, at::ScalarType::Float);
  }
  auto masks_tensors = at::tensor(masks, self.options().dtype(at::kLong));
  OpCommand cmd;
  cmd.Name("IndexPutV2")
      .Input(temp_self, (string)"x")
      .Input(temp_value, (string)"value")
      .Input(masks_tensors, (string)"indexed_sizes")
      .Input(masks_tensors, (string)"indexed_strides");
  for (int i = 0; i < all_defined_indices.size(); i++) {
    string input_name = "indices" +std::to_string(i);
    cmd.Input(all_defined_indices[i], input_name);
  }
  cmd.DynamicInputReg(indexput_func<ge::op::IndexPutV2>, {{all_defined_indices.size(), 4}})
      .Output(result, (string)"x")
      .Attr("_exclude_engines", (string)"AiCore")
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
    std::vector<at::Tensor> all_defined_indices,
    at::SmallVector<int64_t, N> masks,
    const at::Tensor& value,
    bool accumulate) {
  if (!NpuUtils::check_match(&self)) {
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
    result = index_put_aicpu_nocheck(
        contiguous_self, contiguous_self, all_defined_indices, masks, value, accumulate);
  } else {
    result = index_put_aicpu_nocheck(self, self, all_defined_indices, masks, value, accumulate);
  }
  return result;
}

at::Tensor& index_put_aicore(
    at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices,
    std::vector<at::Tensor> all_defined_indices,
    const at::Tensor& value,
    bool accumulate) {
  std::vector<at::Tensor> indices_expand;
  at::SmallVector<int64_t, N> indices_expand_mask;
  at::Tensor value_broadcast;
  auto input_shape = array_to_small_vector(self.sizes());
  c10::List<c10::optional<at::Tensor>> shaped_indices;

  // transfer input and index to 1 dim tensor if index is bool type, and the shape of input and index is the same
  if (all_defined_indices.size() == 1) {
    auto indices_shape = array_to_small_vector(all_defined_indices[0].sizes());
    if (all_defined_indices[0].scalar_type() == at::kBool && input_shape == indices_shape) {
      self = self.reshape(-1);
      shaped_indices.push_back(all_defined_indices[0].reshape(-1));
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
    at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
      index_put_aicore_nocheck(contiguous_self, indices_expand, masks, indices_expand_mask, value_broadcast, accumulate);
      self.copy_(contiguous_self);
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
  std::vector<at::Tensor> all_defined_indices;
  for (c10::optional<at::Tensor> index_opt : indices) {
    if (index_opt.has_value()) {
      const auto& index = *index_opt;
      if (index.defined()) {
        all_defined_indices.emplace_back(index);
        masks.emplace_back(1);
      } else {
        masks.emplace_back(0);
      }
    } else {
      masks.emplace_back(0);
    }
  }
  check_size(value, self, all_defined_indices);
  for (auto &all_defined_indice : all_defined_indices) {
    if (all_defined_indice.device() != self.device()) {
      all_defined_indice = all_defined_indice.to(self.device());
    }
  }

  OpPreparation::CastBackToOriFormat(self);
  at::Tensor value_copy = value;
  at::Tensor self_copy = self;
  OpPreparation::CastBackToOriFormat(value_copy);

  bool aicpu_true = is_aicpu_valid(self, all_defined_indices, masks);
  if (aicpu_true) {
    index_put_aicpu(self_copy, self_copy, all_defined_indices, masks, value_copy, accumulate);
    self.copy_(self_copy);
  } else {
    index_put_aicore(self_copy, indices, all_defined_indices, value_copy, accumulate);
    self.copy_(self_copy);
  }
  return self;
}

} // namespace native
} // namespace at_npu
