#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace at_npu {
namespace native {

class SliceContiguousOpt : public ContiguousOpt {
public:
  bool Optimizer(at::Tensor &self, const at::Tensor &src,
                 const ContiguousTensorDesc &src_desc) override {
    // Pattern slice.
    // Current pattern does not directly depend on other patterns.
    // The relative sequence of this pattern and other patterns is not
    // important.
    c10::SmallVector<int64_t, MAX_DIM> offsets;
    c10::SmallVector<int64_t, MAX_DIM> size;
    if (can_use_slice(src_desc, offsets, size)) {
      RECORD_FUNCTION("contiguous_d_Slice", std::vector<c10::IValue>({src}));
      slice_to_contiguous(self, src, offsets, size, src_desc);
      return true;
    }
    return false;
  }

  bool CanOptimizer(const ContiguousTensorDesc &src_desc) override {
    c10::SmallVector<int64_t, MAX_DIM> offsets;
    c10::SmallVector<int64_t, MAX_DIM> size;
    return can_use_slice(src_desc, offsets, size);
  }

    bool CachedOptimizer(at::Tensor &self, const at::Tensor &src,
                         const ContiguousTensorDesc &src_desc) override
    {
        if (src_desc.cached_contiguous) {
            RECORD_FUNCTION("cached_contiguous_d_Slice", std::vector<c10::IValue>({src}));
            CachedContiguousOpt cachedContiguousOpt = TransContiguous::cached_contiguous_opt[src_desc.hash_src_desc];
            c10::SmallVector<int64_t, MAX_DIM> size = cachedContiguousOpt.cached_opt_parameters.pop_back_val();
            c10::SmallVector<int64_t, MAX_DIM> offsets = cachedContiguousOpt.cached_opt_parameters.pop_back_val();
            slice_to_contiguous(self, src, offsets, size, src_desc);
            return true;
        }
        c10::SmallVector<int64_t, MAX_DIM> offsets;
        c10::SmallVector<int64_t, MAX_DIM> size;
        if (can_use_slice(src_desc, offsets, size)) {
            RECORD_FUNCTION("contiguous_d_Slice", std::vector<c10::IValue>({src}));
            CachedContiguousOpt cached_opt = CachedContiguousOpt{
                    "slice"
            };
            cached_opt.cached_opt_parameters.emplace_back(offsets);
            cached_opt.cached_opt_parameters.emplace_back(size);
            cached_opt.contiguous_tensor_desc = src_desc;
            TransContiguous::cached_contiguous_opt[src_desc.hash_src_desc] = cached_opt;
            slice_to_contiguous(self, src, offsets, size, src_desc);
            return true;
        }
        return false;
    }

private:
  // npu-slice pattern cover several view ops, including chunk, split, narrow
  // and part of index. Judgment logic is based on the implement of view ops in
  // adapter layer.
  bool can_use_slice(const ContiguousTensorDesc &src_desc,
                     c10::SmallVector<int64_t, MAX_DIM> &offsets,
                     c10::SmallVector<int64_t, MAX_DIM> &size) {
    const auto &base_sizes = src_desc.base_sizes_;
    const auto &base_strides = src_desc.base_strides_;
    auto view_sizes = src_desc.sizes_;
    auto view_strides = src_desc.strides_;

    // narrow+select(select at last dim) ==> single narrow
    // 限制条件：1. 最后一轴stride非1==>最后一轴select；2.
    // 基础格式；3.非最后一轴发生narrow（元素减少）
    // 最小化影响：仅限最后一轴的select，即tensor.select(-1, 1) ==
    // tensor[**,1:2],select过渡到narrow
    if (view_strides[view_strides.size() - 1] != 1 &&
        FormatHelper::IsBaseFormatType(src_desc.npu_format_) &&
        view_strides.size() < base_strides.size() &&
        c10::multiply_integers(view_sizes) <
            c10::multiply_integers(base_sizes) / base_sizes[base_sizes.size() - 1]) {
      view_sizes.emplace_back(1);
      view_strides.emplace_back(1);
    }

    // Strides must be the same.
    if (view_strides != base_strides) {
      return false;
    }

    // Only narrow dims are different.
    c10::SmallVector<int64_t, MAX_DIM> narrow_dims;
    if (view_sizes.size() != base_sizes.size()) {
      return false;
    }
    for (const auto i : c10::irange(view_sizes.size())) {
      if (view_sizes[i] == base_sizes[i]) {
        narrow_dims.emplace_back(0);
      } else if (view_sizes[i] < base_sizes[i]) {
        narrow_dims.emplace_back(1);
      } else {
        return false;
      }
    }

    // Calculate npu slice param.
    size = view_sizes;
    offsets.clear();
    int64_t storage_offsets = src_desc.offset_;
    // src.storage_offset() == start[narrow_dims[i]]*stride[narrow_dims[i]]
    for (const auto i : c10::irange(view_strides.size())) {
      offsets.emplace_back(storage_offsets / view_strides[i]);
      storage_offsets = storage_offsets % view_strides[i];
    }
    if (storage_offsets != 0) {
      return false;
    }
    for (const auto i : c10::irange(offsets.size())) {
      if ((offsets[i] + view_sizes[i]) > base_sizes[i]) {
        // In narrow calculation, (start + length) <= cur_size
        return false;
      }
      if (offsets[i] != 0 && narrow_dims[i] == 0) {
        // narrow_dims[i] == 0 means dim i is not involved in narrow
        // calculation. offsets[i] != 0 means dim i has the start of narrow
        // calculation. Two conditions are contradictory.
        return false;
      }
    }
    return true;
  }


    void slice_to_contiguous(at::Tensor &self, const at::Tensor &src,
                             const c10::SmallVector<int64_t, MAX_DIM> &offsets,
                             const c10::SmallVector<int64_t, MAX_DIM> &size,
                             const ContiguousTensorDesc &src_desc) {
        // create contiguous tensor for npu slice
        const auto &temp_tensor_size = src_desc.base_sizes_;
        at::Tensor temp_src = TransContiguous::view_tensor(src, src_desc.base_offset_, temp_tensor_size, src_desc.base_strides_);

        custom_ops::npu_slice_out(temp_src, offsets, size, self);
        return;
    }
}; // class SliceContiguousOpt

REGISTER_COPY_OPT(slice, SliceContiguousOpt)

} // namespace native
} // namespace at_npu