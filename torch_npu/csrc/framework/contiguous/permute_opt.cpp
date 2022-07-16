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

#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

class PermuteContiguousOpt : public ContiguousOpt {
public:
  bool Optimizer(at::Tensor &self, const at::Tensor &src,
                 const ContiguousTensorDesc &src_desc) override {
    // pattern permute
    c10::SmallVector<int64_t, MAX_DIM> perm;
    c10::SmallVector<int64_t, 5> sizes;
    if (can_use_permute(src_desc, perm, sizes)) {
      RECORD_FUNCTION("npuTranspose", std::vector<c10::IValue>({src}));
      // Refresh src Tensor to match output self Tensor
      auto src_desc_stored = torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc();
      auto &src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
      src_desc.base_sizes_ = sizes;
      src_desc.base_strides_ = StorageDescHelper::ComputeStrideFromShape(
          static_cast<FormatShape>(sizes));
      src_desc.storage_sizes_ = sizes;

      NPUNativeFunctions::npu_transpose_out(src, perm, false, self);
      src_desc = src_desc_stored;
      return true;
    }
    return false;
  }

  bool CanOptimizer(const ContiguousTensorDesc &src_desc) override {
    c10::SmallVector<int64_t, MAX_DIM> perm;
    c10::SmallVector<int64_t, 5> sizes;
    return can_use_permute(src_desc, perm, sizes);
  }

private:
  bool can_use_permute(const ContiguousTensorDesc &src_desc,
                       c10::SmallVector<int64_t, MAX_DIM> &perm,
                       c10::SmallVector<int64_t, 5> &sizes) {
    const auto &base_sizes = src_desc.base_sizes_;
    const auto &base_strides = src_desc.base_strides_;
    auto view_sizes = src_desc.sizes_;
    auto view_strides = src_desc.strides_;

    c10::SmallVector<int64_t, MAX_DIM> indexes;
    for (auto i = 0; i < src_desc.sizes_.size(); i++) {
      indexes.emplace_back(i);
    }

    // After permute or reshape+permute, the total amount of data remains
    // unchanged.
    if (at::prod_intlist(view_sizes) != at::prod_intlist(base_sizes)) {
      return false;
    }

    // Reorder axes of shape and stride in descending order
    for (auto i = 0; i < src_desc.sizes_.size() - 1; i++) {
      for (auto j = i + 1; j < src_desc.sizes_.size(); j++) {
        bool need_swap = (view_strides[i] < view_strides[j]) ||
                         (view_strides[i] == view_strides[j] &&
                          view_sizes[i] < view_sizes[j]);
        if (need_swap) {
          std::swap(view_strides[i], view_strides[j]);
          std::swap(view_sizes[i], view_sizes[j]);
          std::swap(indexes[i], indexes[j]);
        }
      }
    }

    // After reordering, check whether the shape and stride match
    auto current_stride = 1;
    for (int64_t i = src_desc.sizes_.size() - 1; i >= 0; i--) {
      if (current_stride != view_strides[i]) {
        NPU_LOGD("After reordering, shape and stride still do not match, and "
                 "permute pattern cannot be used.");
        return false;
      }
      current_stride *= view_sizes[i];
    }
    if ((base_sizes.size() - view_sizes.size()) !=
        (base_strides.size() - view_strides.size())) {
      NPU_LOGD("Reordered shape and base shape do not match, and permute "
               "pattern cannot be used.");
      return false;
    }

    if (c10_npu::NpuRunMode::IsGraphMode()) {
      // Could be permute or squeeze/unsqueeze + permute
      auto view_sizes_squeeze = view_sizes;
      auto view_strides_squeeze = view_strides;
      squeeze_shape_and_stride(view_sizes_squeeze, view_strides_squeeze);
      auto base_sizes_squeeze = base_sizes;
      auto base_strides_squeeze = base_strides;
      squeeze_shape_and_stride(base_sizes_squeeze, base_strides_squeeze);
      bool dim_equal =
          (view_sizes_squeeze.size() == base_sizes_squeeze.size()) &&
          (view_strides_squeeze.size() == base_strides_squeeze.size());
      if (!dim_equal) {
        NPU_LOGD(
            "After squeezing, reordered shape and base shape do not match, "
            "and permute pattern cannot be used.");
        return false;
      }
      for (auto i = 0; i < view_sizes_squeeze.size(); i++) {
        if ((view_sizes_squeeze[i] != base_sizes_squeeze[i]) ||
            (view_strides_squeeze[i]) != base_strides_squeeze[i]) {
          NPU_LOGD("After squeezing, reordered shape and base shape do not "
                   "match, and permute pattern cannot be used.");
          return false;
        }
      }
    }

    // Calculate perm and sizes for permute
    for (const auto ele : view_sizes) {
      sizes.emplace_back(ele);
    }
    perm = indexes;
    for (int64_t i = 0; i < src_desc.sizes_.size(); i++) {
      perm[indexes[i]] = i;
    }
    return true;
  }

  void optimize_permute(c10::SmallVector<int64_t, MAX_DIM> &perm,
                        c10::SmallVector<int64_t, 5> &sizes) {
    c10::SmallVector<int64_t, MAX_DIM> optimized_perm;
    c10::SmallVector<int64_t, 5> optimized_sizes;
    if (perm.size() != sizes.size()) {
      NPU_LOGD("Param perm and sizes do not match.");
      return;
    }

    // Gather index
    for (auto i = 0; i < perm.size(); i++) {
      auto temp_perm_i = perm[i];
      auto temp_sizes_i = sizes[perm[i]];
      for (auto j = i + 1; j < perm.size(); j++) {
        if (perm[i] + 1 == perm[j]) {
          temp_sizes_i *= sizes[perm[j]];
          ++i;
          continue;
        }
        break;
      }
      if (temp_sizes_i == 1) {
        // Optimize permute calculation for better performance, by squeezing
        // permute param.
        continue;
      }
      optimized_perm.emplace_back(temp_perm_i);
      optimized_sizes.emplace_back(temp_sizes_i);
    }
    if (optimized_perm.size() == perm.size()) {
      NPU_LOGD("No adjacent axes, cannot be optimized.");
      return;
    }

    // Calculate new perm and shape
    c10::SmallVector<int64_t, MAX_DIM> perm_indexes;
    for (auto i = 0; i < optimized_perm.size(); i++) {
      perm_indexes.emplace_back(i);
    }
    for (auto i = 0; i < optimized_perm.size() - 1; i++) {
      for (auto j = i + 1; j < optimized_perm.size(); j++) {
        if (optimized_perm[i] > optimized_perm[j]) {
          std::swap(optimized_perm[i], optimized_perm[j]);
          std::swap(perm_indexes[i], perm_indexes[j]);
        }
      }
    }
    perm = perm_indexes;
    for (auto i = 0; i < perm_indexes.size(); i++) {
      perm[perm_indexes[i]] = i;
    }
    sizes = optimized_sizes;
    for (auto i = 0; i < perm_indexes.size(); i++) {
      sizes[i] = optimized_sizes[perm_indexes[i]];
    }
  }

  template <typename T> void squeeze_shape_and_stride(T &shape, T &stride) {
    IF_GRAPH_MODE_THEN_RUN(return;)
    auto shape_size = shape.size();
    for (auto i = 0; i < shape_size; i++) {
      if (shape[i] == 1) {
        shape.erase(shape.begin() + i);
        stride.erase(stride.begin() + i);
        --i;
      }
    }
  }

}; // class PermuteContiguousOpt

REGISTER_COPY_OPT(permute, PermuteContiguousOpt)

} // namespace native
} // namespace at_npu