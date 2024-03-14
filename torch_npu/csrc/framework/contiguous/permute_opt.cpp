#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "op_plugin/OpInterface.h"

namespace at_npu {
namespace native {

class PermuteContiguousOpt : public ContiguousOpt {
public:
    bool Optimizer(at::Tensor &self, const at::Tensor &src,
                   const ContiguousTensorDesc &src_desc) override
    {
        // pattern permute
        c10::SmallVector<int64_t, MAX_DIM> perm;
        c10::SmallVector<int64_t, 5> sizes;
        if (can_use_permute(src_desc, perm, sizes)) {
            RECORD_FUNCTION("contiguous_d_Transpose", std::vector<c10::IValue>({src}));
            permute_to_contiguous(self, src, perm, sizes);
            return true;
        }
        return false;
    }

  bool CanOptimizer(const ContiguousTensorDesc &src_desc) override {
    c10::SmallVector<int64_t, MAX_DIM> perm;
    c10::SmallVector<int64_t, 5> sizes;
    return can_use_permute(src_desc, perm, sizes);
  }

    bool CachedOptimizer(at::Tensor &self, const at::Tensor &src,
                         const ContiguousTensorDesc &src_desc) override
    {
        if (src_desc.cached_contiguous) {
            RECORD_FUNCTION("cached_contiguous_d_Transpose", std::vector<c10::IValue>({src}));
            CachedContiguousOpt cachedContiguousOpt = TransContiguous::cached_contiguous_opt[src_desc.hash_src_desc];
            c10::SmallVector<int64_t, MAX_DIM> sizes = cachedContiguousOpt.cached_opt_parameters.pop_back_val();
            c10::SmallVector<int64_t, MAX_DIM> perm = cachedContiguousOpt.cached_opt_parameters.pop_back_val();
            permute_to_contiguous(self, src, perm, sizes);
            return true;
        }

        // pattern permute
        c10::SmallVector<int64_t, MAX_DIM> perm;
        c10::SmallVector<int64_t, MAX_DIM> sizes;
        if (can_use_permute(src_desc, perm, sizes)) {
            RECORD_FUNCTION("contiguous_d_Transpose", std::vector<c10::IValue>({src}));
            CachedContiguousOpt cached_opt = CachedContiguousOpt{
                    "permute"
            };
            cached_opt.cached_opt_parameters.emplace_back(perm);
            cached_opt.cached_opt_parameters.emplace_back(sizes);
            cached_opt.contiguous_tensor_desc = src_desc;
            TransContiguous::cached_contiguous_opt[src_desc.hash_src_desc] = cached_opt;
            permute_to_contiguous(self, src, perm, sizes);
            return true;
        }
        return false;
    }

private:

    void permute_to_contiguous(at::Tensor &self, const at::Tensor &src,
                               const c10::SmallVector<int64_t, MAX_DIM> &perm,
                               const c10::SmallVector<int64_t, MAX_DIM> &sizes)
    {
        // Refresh src Tensor to match output self Tensor
        auto src_desc_stored = torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc();
        auto &src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
        src_desc.base_sizes_ = sizes;
        src_desc.base_strides_ = StorageDescHelper::ComputeStrideFromShape(static_cast<FormatShape>(sizes));
        src_desc.storage_sizes_ = sizes;
        op_plugin::npu_transpose_out(src, perm, false, self);
        src_desc = src_desc_stored;
    }

    bool can_use_permute(const ContiguousTensorDesc &src_desc,
                         c10::SmallVector<int64_t, MAX_DIM> &perm,
                         c10::SmallVector<int64_t, 5> &sizes)
    {
        const auto &base_sizes = src_desc.base_sizes_;
        const auto &base_strides = src_desc.base_strides_;
        auto view_sizes = src_desc.sizes_;
        auto view_strides = src_desc.strides_;

        c10::SmallVector<int64_t, MAX_DIM> indexes;
        for (const auto i : c10::irange(src_desc.sizes_.size())) {
            indexes.emplace_back(i);
        }

        // After permute or reshape+permute, the total amount of data remains
        // unchanged.
        if (c10::multiply_integers(view_sizes) != c10::multiply_integers(base_sizes)) {
            return false;
        }

        // Reorder axes of shape and stride in descending order
        for (const auto i : c10::irange(src_desc.sizes_.size() - 1)) {
            for (const auto j : c10::irange(i + 1, src_desc.sizes_.size())) {
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
        int64_t src_desc_sizes = static_cast<int64_t>(src_desc.sizes_.size());
        for (int64_t i = src_desc_sizes - 1; i >= 0; i--) {
            if (current_stride != view_strides[i]) {
                ASCEND_LOGD("After reordering, shape and stride still do not match, and "
                            "permute pattern cannot be used.");
                return false;
            }
            current_stride *= view_sizes[i];
        }
        if ((base_sizes.size() - view_sizes.size()) !=
            (base_strides.size() - view_strides.size())) {
            ASCEND_LOGD("Reordered shape and base shape do not match, and permute "
                        "pattern cannot be used.");
            return false;
        }

        // Calculate perm and sizes for permute
        for (const auto ele : view_sizes) {
            sizes.emplace_back(ele);
        }
        perm = indexes;
        for (const auto i : c10::irange(src_desc.sizes_.size())) {
            perm[indexes[i]] = i;
        }
        return true;
    }

  void optimize_permute(c10::SmallVector<int64_t, MAX_DIM> &perm,
                        c10::SmallVector<int64_t, 5> &sizes) {
    c10::SmallVector<int64_t, MAX_DIM> optimized_perm;
    c10::SmallVector<int64_t, 5> optimized_sizes;
    if (perm.size() != sizes.size()) {
      ASCEND_LOGD("Param perm and sizes do not match.");
      return;
    }

    // Gather index
    int64_t perm_size = static_cast<int64_t>(perm.size());
    for (int64_t i = 0; i < perm_size; i++) {
      auto temp_perm_i = perm[i];
      auto temp_sizes_i = sizes[perm[i]];
      for (const auto j : c10::irange(i + 1, perm_size)) {
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
      ASCEND_LOGD("No adjacent axes, cannot be optimized.");
      return;
    }

    // Calculate new perm and shape
    c10::SmallVector<int64_t, MAX_DIM> perm_indexes;
    for (const auto i : c10::irange(optimized_perm.size())) {
      perm_indexes.emplace_back(i);
    }
    for (const auto i : c10::irange(optimized_perm.size() - 1)) {
      for (const auto j : c10::irange(i + 1, optimized_perm.size())) {
        if (optimized_perm[i] > optimized_perm[j]) {
          std::swap(optimized_perm[i], optimized_perm[j]);
          std::swap(perm_indexes[i], perm_indexes[j]);
        }
      }
    }
    perm = perm_indexes;
    for (const auto i : c10::irange(perm_indexes.size())) {
      perm[perm_indexes[i]] = i;
    }
    sizes = optimized_sizes;
    for (const auto i : c10::irange(perm_indexes.size())) {
      sizes[i] = optimized_sizes[perm_indexes[i]];
    }
  }

  template <typename T> void squeeze_shape_and_stride(T &shape, T &stride) {
    int64_t shape_size = static_cast<int64_t>(shape.size());
    for (int64_t i = 0; i < shape_size; i++) {
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