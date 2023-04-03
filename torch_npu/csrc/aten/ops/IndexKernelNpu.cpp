// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

#include <third_party/acl/inc/op_proto/index_ops.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/AdvancedIndex.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/graph/util/GraphModeGuard.h"

namespace at_npu {
namespace native {
namespace {
DynamicInputRegFunc index_func =
  [](DyNumAndIndex num_and_index, std::string op_name) -> ge::OperatorPtr {
    auto ge_op = std::make_shared<ge::op::Index>(op_name.c_str());
    ge_op->create_dynamic_input_byindex_indices(
        num_and_index.front().first, num_and_index.front().second);
    return ge_op;
  };
}

// Limitations of the aicore branch
bool check_index_aicore(const at::Tensor& self, const at::TensorList& indices, const at::IntArrayRef masks, const at::Tensor& result) {
  // The bool index only supports the input of 2D and 3D.
  if ((self.dim() == 1 && indices[0].scalar_type() == at::kBool) ||
      (self.dim() > 3 && indices[0].scalar_type() == at::kBool)) {
    return false;
  }
  // Relax the scene : the dtype of indices is bool, x'shape is (n, m) and indices's shape is (m,).
  // When the dtype of indices is bool, if the number of outputs exceeds 5000, go to AICPU.
  if (self.dim() == 2 && indices[0].scalar_type() == at::kBool && masks.size() == 2 && indices.size() == 1 &&
      result.numel() < 5000) {
    return true;
  }

  // The input of the aicore does not support float64.
  // Indices should start from dimension 0 of x and continuous.
  if (self.scalar_type() == at::kDouble || masks[0] == 0 || masks.size() > indices.size()) {
    return false;
  }

  if (indices.size() > 1) {
    // The dtype of indices only support int64, and all indices's shape is (n,).
    if (indices[0].scalar_type() != at::kLong || indices[0].dim() != 1) {
      return false;
    }
    for (int32_t idx = 1; idx < indices.size(); idx++) {
      if (indices[idx].scalar_type() != at::kLong ||
          indices[idx].dim() != 1 ||
          indices[idx].sizes() != indices[idx - 1].sizes()) {
        return false;
      }
    }
    if (self.dim() == indices.size()) {
      return true;
    }
  }

  if (indices.size() < 2) {
    // The dtype of indices can only be int64 or bool.
    if (indices[0].scalar_type() != at::kLong && indices[0].scalar_type() != at::kBool) {
      return false;
    }
    // When the dtype of indices is bool, if the number of outputs exceeds 5000, go to AICPU.
    if (indices[0].scalar_type() == at::kBool && result.numel() < 5000) {
      // when x'shape is (n, 1), there exist abnormal execution in the model.
      if (self.dim() == 2 && self.size(1) == 1) {
        return false;
      }
      // Relax the scene : the dtype of indices is bool, and x'shape is equal to indices's shape.
      // Relax the scene : the dtype of indices is bool, and x'shape is (n, m) and indices's shape is (n,) and m > 1.
      if (indices[0].sizes() == self.sizes() ||
          (self.dim() == 2 && indices[0].size(0) == self.size(0) && indices[0].dim() == 1)) {
        return true;
      }
    } else if (indices[0].scalar_type() == at::kLong && indices[0].dim() == 1) {
      // Relax the scene : the dtype of indices is int64, the shape of the indices is 1d.
      return true;
    }
  }
  return false;
}

at::Tensor& index_out_nocheck_npu(
    const at::Tensor& self,
    const at::IntArrayRef masks,
    const at::TensorList& indices,
    at::Tensor& result) {
  bool is_aicore = check_index_aicore(self, indices, masks, result);
  OpCommand cmd;
  if (!is_aicore) { 
    cmd.Name("Index")
        .Input(self)
        .Input(masks, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Input(result.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT);
    for (int i = 0; i < indices.size(); i++) {
      std::string name = "indices" + std::to_string(i);
      cmd.Input(indices[i], name);
    }
    cmd.Output(result)
        .Attr("_exclude_engines", (string)"AiCore")
        .Run();
  } else {
    if (indices.size() > 1) {
      at::Tensor make_sizes_tensor = at::tensor(masks, self.options().dtype(at::kLong));
      at::Tensor make_strides_tensor = at::tensor(result.sizes(), self.options().dtype(at::kLong));
      cmd.Name("Index")
          .Input(self, (string)"x")
          .Input(make_sizes_tensor, (string)"indexed_sizes")
          .Input(make_strides_tensor, (string)"indexed_strides");
      for (int i = 0; i < indices.size(); i++) {
        std::string name = "indices" + std::to_string(i);
        cmd.Input(indices[i], name);
      }
      cmd.DynamicInputReg(index_func, {{indices.size(), 3}})
          .Output(result)
          .Run();
    } else {
      cmd.Name("Index")
          .Input(self)
          .Input(masks, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
          .Input(result.sizes(), at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT);
      if (indices[0].scalar_type() == at::kBool && masks.size() == 2 && indices.size() == 1) {
        // For the scene : x'shape is (n, m) and indices's shape is (m,),
        // and turn indices's shape to be (n, m).
        at::Tensor tmp_tensor = indices[0].unsqueeze(0);
        c10::SmallVector<int64_t, N> self_size = array_to_small_vector(self.sizes());
        at::Tensor full_indices = NPUNativeFunctions::npu_broadcast(tmp_tensor, self_size);
        cmd.Input(full_indices, "indices0");
      } else if (indices[0].scalar_type() == at::kBool && indices[0].sizes() != self.sizes()) {
        // For the scene : x'shape is (n, m) and indices's shape is (n,),
        // and turn indices's shape to be (n, m).
        at::Tensor tmp_tensor = indices[0].unsqueeze(1);
        c10::SmallVector<int64_t, N> self_size = array_to_small_vector(self.sizes());
        at::Tensor full_indices = NPUNativeFunctions::npu_broadcast(tmp_tensor, self_size);
        cmd.Input(full_indices, "indices0");
      } else {
        cmd.Input(indices[0], "indices0");
      }
      cmd.Output(result)
         .Run();
    }
  }
  return result;
}

at::Tensor index_high_dims(const at::Tensor& self, std::vector<at::Tensor> indices) {
  // masks corresponds to indices. 0 indicates undefined tensor.
  at::SmallVector<int64_t, N> masks;
  std::vector<at::Tensor> all_defined_indices;
  for (int i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      all_defined_indices.emplace_back(indices[i]);
      masks.emplace_back(1);
      continue;
    }
    masks.emplace_back(0);
  }

  /**
   * When input.size(0) = 1, if the dtype of indices is int64,
   * and indices only for 0 dimension, can broadcast to output.
   */
  if (self.size(0) == 1 && masks.size() == 1 && masks[0] == 1 &&
      all_defined_indices[0].scalar_type() == at::kLong && all_defined_indices[0].dim() == 1) {
    c10::SmallVector<int64_t, N> output_size = array_to_small_vector(self.sizes());
    output_size[0] = all_defined_indices[0].size(0);
    at::Tensor result = NPUNativeFunctions::npu_broadcast(self, output_size);
    return result;
  }

  at::Tensor self_nd = NPUNativeFunctions::npu_format_cast(self, ACL_FORMAT_ND);

  auto output_size = index_npu_output_size(self_nd, indices);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(self_nd, output_size, ACL_FORMAT_ND);

  // calculate the output result of the NPU
  index_out_nocheck_npu(self_nd, masks, all_defined_indices, result);

  return result;
}

at::Tensor NPUNativeFunctions::index(const at::Tensor& self, const torch::List<c10::optional<at::Tensor>>& orig) {
  /**
   * In the cann framework, index operator belongs to the fourth type of
   * operator, which means that the execution of the index operator must go
   * through the dynamic shape execution framework. In this case, constructing
   * a large dynamic shape graph is not beneficial to the overall execution
   * performance, because more dynamic shape operators are introduced.
   * Therefore, when the fourth type of operator is encountered in graph
   * mode, the single op mode is switched to execute by default.
   */
  if (self.device().type() == at::kCPU) {
    return at::native::index(self, orig);
  }

  GraphModeGuard mode_guard(c10_npu::ModeKind::SINGLE_OP_MODE);

  at::native::checkIndexTensorTypes(orig);
  auto indices = AdvanceIndex::npu_expand_tensors(self, orig);

  // not to transpose at non-binary scene
  if (!env::CheckJitDisable()) {
    return index_high_dims(self, indices);
  }

  // masks corresponds to indices. 0 indicates undefined tensor.
  at::SmallVector<int64_t, N> masks;
  std::vector<at::Tensor> all_defined_indices;

  /**
   * indices_flag: 0, 1, 2, 3
   * 0 -- dim 0 and contiguous, e.g. masks: [1, 1, 1]
   * 1 -- not dim 0 and contiguous, e.g. masks: [0, 1, 1, 1] or [0, 0, 1]
   * 2 -- not contiguous, e.g. masks: [0, 1, 1, 0, 1] or [1, 1, 0, 1]
   * 3 -- index.dim > 0, e.g. indices: [:, [[1, 0], [0, 1]]]
   */
  int indices_flag = 0;
  int is_1_in_masks = 0;
  for (int i = 0; i < indices.size(); i++) {
    if (indices[i].dim() > 1) {
      indices_flag = 3;
      break;
    }
    if (indices[i].defined()) {
      all_defined_indices.emplace_back(indices[i]);
      masks.emplace_back(1);
      is_1_in_masks = 1;
      continue;
    }
    masks.emplace_back(0);
    if (indices_flag == 2) {
      continue;
    }
    if (is_1_in_masks == 1) {
      indices_flag = 2;
    } else {
      indices_flag = 1;
    }
  }
  if (indices_flag == 3) {
    return index_high_dims(self, indices);
  }

  /**
   * When input.size(0) = 1, if the dtype of indices is int64,
   * and indices only for 0 dimension, can broadcast to output.
   */
  if (self.size(0) == 1 && masks.size() == 1 && masks[0] == 1 &&
      all_defined_indices[0].scalar_type() == at::kLong && all_defined_indices[0].dim() == 1) {
    c10::SmallVector<int64_t, N> output_size = array_to_small_vector(self.sizes());
    output_size[0] = all_defined_indices[0].size(0);
    at::Tensor result = NPUNativeFunctions::npu_broadcast(self, output_size);
    return result;
  }

  at::Tensor self_nd = NPUNativeFunctions::npu_format_cast(self, ACL_FORMAT_ND);

  at::SmallVector<int64_t, N> masks_trans = {};
  if (indices_flag != 0) {
    int index_num = all_defined_indices.size();
    masks_trans.assign(index_num, 1);
  } else {
    masks_trans = masks;
  }

  at::SmallVector<int64_t, N> perm_0 = {};
  at::SmallVector<int64_t, N> perm_1 = {};
  for (int i = 0; i < masks.size(); i++) {
    if (masks[i] == 0) {
      perm_0.emplace_back(i);
    } else {
      perm_1.emplace_back(i);
    }
  }
  perm_1.insert(perm_1.end(), perm_0.begin(), perm_0.end());
  int supplement_iter = self_nd.dim() - perm_1.size();
  int perm_1_size = perm_1.size();
  if (perm_1.size() < self_nd.dim()) {
    for (int i = 0; i < supplement_iter; i++) {
      perm_1.emplace_back(perm_1_size + i);
    }
  }

  at::Tensor self_nd_trans = (perm_0.size() != 0) ?
      NPUNativeFunctions::npu_transpose(self_nd, perm_1, true) : self_nd;

  auto output_size = index_npu_output_size(self_nd_trans, all_defined_indices);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(self_nd_trans, output_size, ACL_FORMAT_ND);

  // calculate the output result of the NPU
  index_out_nocheck_npu(self_nd_trans, masks_trans, all_defined_indices, result);

  if (indices_flag == 1) {
    int out_dim = result.dim();
    at::SmallVector<int64_t, N> perm_flag1 = {};
    for (int i = 1; i < out_dim; i++) {
      perm_flag1.emplace_back(i);
    }
    perm_flag1.insert(perm_flag1.begin() + perm_1[0], 0);
    result = NPUNativeFunctions::npu_transpose(result, perm_flag1, true);
  }

  return result;
}

} // namespace native
} // namespace at_npu
