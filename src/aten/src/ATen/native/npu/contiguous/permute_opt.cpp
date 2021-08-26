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

#include <ATen/native/npu/contiguous/ContiguousOpt.h>
#include <ATen/native/npu/utils/KernelNpuOutputSize.h>

namespace at {
namespace native {
namespace npu {

class PermuteContiguousOpt : public ContiguousOpt {
public:  
  bool Optimizer(const Tensor& src, Tensor& self) override {
    // pattern permute
    SmallVector<int64_t, SHAPE_SIZE> perm;
    SmallVector<int64_t, 5> sizes;
    if (can_use_permute(src, perm, sizes)) {
      //TODO(ascend): delete call and implementation, after more test 
      // optimize_permute(perm, sizes);
      RECORD_FUNCTION("npuTransposeD", std::vector<c10::IValue>({src}));
      // create contiguous tensor for npu transpose
      Tensor temp_src = at::empty(sizes, src.options());
      temp_src.set_(src.storage(), temp_src.storage_offset(), temp_src.sizes(), temp_src.strides());
      auto npu_desc = temp_src.storage().unsafeGetStorageImpl()->npu_desc_;
      temp_src.storage().unsafeGetStorageImpl()->npu_desc_.base_sizes_ = temp_src.sizes();
      temp_src.storage().unsafeGetStorageImpl()->npu_desc_.base_strides_ = temp_src.strides();
      temp_src.storage().unsafeGetStorageImpl()->npu_desc_.storage_sizes_ = temp_src.sizes();

      at::npu_transpose_out(self, temp_src, perm);
      temp_src.storage().unsafeGetStorageImpl()->npu_desc_ = npu_desc;
      return true;
    }
    return false;
  }

  bool CanOptimizer(const Tensor& src) override {
    SmallVector<int64_t, SHAPE_SIZE> perm;
    SmallVector<int64_t, 5> sizes;
    return can_use_permute(src, perm, sizes);
  }
  
private:
  bool can_use_permute(const Tensor &src, 
      SmallVector<int64_t, SHAPE_SIZE> &perm, 
      SmallVector<int64_t, 5> &sizes) {
    // uncontiguous
    if (src.is_contiguous()) {
        return false;
    }

    auto base_sizes = src.storage().get_npu_desc().base_sizes_;
    auto base_strides = src.storage().get_npu_desc().base_strides_;
    auto view_sizes = array_to_small_vector(src.sizes());
    auto view_strides = array_to_small_vector(src.strides());
    SmallVector<int64_t, SHAPE_SIZE> indexes;
    for (auto i = 0; i < src.dim(); i++) {
        indexes.emplace_back(i);
    }

    // Reorder axes of shape and stride in descending order
    for (auto i = 0; i < src.dim() - 1; i++) {
        for (auto j = i + 1; j < src.dim(); j++) {
        bool need_swap = (view_strides[i] < view_strides[j]) ||
                        (view_strides[i] == view_strides[j] && view_sizes[i] < view_sizes[j]);
        if (need_swap) {
            std::swap(view_strides[i], view_strides[j]);
            std::swap(view_sizes[i], view_sizes[j]);
            std::swap(indexes[i], indexes[j]);
        }
        }
    }

    // After reordering, check whether the shape and stride match
    auto current_stride = 1;
    for (int64_t i = src.dim() - 1; i >= 0; i--) {
        if (current_stride != view_strides[i]) {
        NPU_LOGD("After reordering, shape and stride still do not match, and permute pattern cannot be used.");
        return false;
        }
        current_stride *= view_sizes[i];
    }
    if ((base_sizes.size() - view_sizes.size()) != (base_strides.size() - view_strides.size())) {
        NPU_LOGD("Reordered shape and base shape do not match, and permute pattern cannot be used.");
        return false;
    }

    // Could be permute or squeeze/unsqueeze + permute
    auto view_sizes_squeeze = view_sizes;
    auto view_strides_squeeze = view_strides;
    squeeze_shape_and_stride(view_sizes_squeeze, view_strides_squeeze);
    auto base_sizes_squeeze = base_sizes;
    auto base_strides_squeeze = base_strides;
    squeeze_shape_and_stride(base_sizes_squeeze, base_strides_squeeze);
    bool dim_equal = (view_sizes_squeeze.size() == base_sizes_squeeze.size()) &&
                    (view_strides_squeeze.size() == base_strides_squeeze.size());
    if (!dim_equal) {
        NPU_LOGD("After squeezing, reordered shape and base shape do not match, and permute pattern cannot be used.");
        return false;
    }
    for (auto i = 0; i < view_sizes_squeeze.size(); i++) {
        if ((view_sizes_squeeze[i] != base_sizes_squeeze[i]) || (view_strides_squeeze[i]) != base_strides_squeeze[i]) {
        NPU_LOGD("After squeezing, reordered shape and base shape do not match, and permute pattern cannot be used.");
        return false;
        }
    }

    // Calculate perm and sizes for permute
    for (const auto ele : view_sizes) {
        sizes.emplace_back(ele);
    }
    perm = indexes;
    for (int64_t i = 0; i < src.dim(); i++) {
        perm[indexes[i]] = i;
    }
    return true;
  }

  
void optimize_permute(SmallVector<int64_t, SHAPE_SIZE> &perm, SmallVector<int64_t, 5> &sizes) {
  SmallVector<int64_t, SHAPE_SIZE> optimized_perm;
  SmallVector<int64_t, 5> optimized_sizes;
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
      // Optimize permute calculation for better performance, by squeezing permute param.
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
  SmallVector<int64_t, SHAPE_SIZE> perm_indexes;
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

template <typename T>
void squeeze_shape_and_stride(T &shape, T &stride) {
  for (auto i = 0; i < shape.size(); i++) {
    if (shape[i] == 1) {
      shape.erase(shape.begin() + i);
      stride.erase(stride.begin() + i);
      --i;
    }
  }
}

}; // class PermuteContiguousOpt

REGISTER_COPY_OPT(permute, PermuteContiguousOpt)

} // npu
} // native
} // at
