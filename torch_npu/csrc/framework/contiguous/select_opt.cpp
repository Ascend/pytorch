#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace at_npu {
namespace native {

class SelectContiguousOpt : public ContiguousOpt {
public:
  bool Optimizer(at::Tensor &self, const at::Tensor &src,
                 const ContiguousTensorDesc &src_desc) override {
    // select(dim, start), length[dim] == 1
    c10::SmallVector<int64_t, MAX_DIM> start;
    c10::SmallVector<int64_t, MAX_DIM> length;

    if (can_use_select(src_desc, start, length)) {
      RECORD_FUNCTION("contiguous_d_StridedSlice",
                      std::vector<c10::IValue>({src}));
      select_to_contiguous(self, src, start, length, src_desc);
      return true;
    }
    return false;
  }

  bool CanOptimizer(const ContiguousTensorDesc &src_desc) override {
    c10::SmallVector<int64_t, MAX_DIM> start;
    c10::SmallVector<int64_t, MAX_DIM> length;
    return can_use_select(src_desc, start, length);
  }

private:
  bool can_use_select(const ContiguousTensorDesc &src_desc,
                      c10::SmallVector<int64_t, MAX_DIM> &start,
                      c10::SmallVector<int64_t, MAX_DIM> &length) {
    // base info and src info
    const auto &base_size = src_desc.base_sizes_;
    const auto &base_stride = src_desc.base_strides_;
    const auto &select_size = src_desc.sizes_;
    const auto &select_stride = src_desc.strides_;

    // len(base_size) - len(select_size) == 1  && len(base_stride) -
    // len(select_stride) == 1
    if ((base_size.size() - select_size.size() != 1) ||
        (base_stride.size() - select_stride.size() != 1)) {
      return false;
    }

    // recover src tensor info: shape and stride
    c10::SmallVector<int64_t, MAX_DIM> temp_size;
    c10::SmallVector<int64_t, MAX_DIM> temp_stride;
    for (size_t i = 0U; i <= select_size.size(); i++) {
      if (base_size[i] != select_size[i] ||
          base_stride[i] != select_stride[i]) {
        temp_size.emplace_back(base_size[i]);
        temp_stride.emplace_back(base_stride[i]);
        for (const auto j : c10::irange(i + 1, select_size.size() + 1)) {
          temp_size.emplace_back(select_size[j - 1]);
          temp_stride.emplace_back(select_stride[j - 1]);
          i = j + 1;
        }
      } else {
        temp_size.emplace_back(select_size[i]);
        temp_stride.emplace_back(select_stride[i]);
      }
    }

    for (const auto i : c10::irange(select_size.size() + 1)) {
      if (base_size[i] == temp_size[i] && base_stride[i] == temp_stride[i]) {
        continue;
      } else {
        return false;
      }
    }

    // Collect the select infos for SliceD: dim, start, length
    // confirm the selected dim
    int64_t dim = static_cast<int64_t>(base_size.size()) - 1;
    for (const auto i : c10::irange(select_size.size())) {
      if (base_size[i] != select_size[i] ||
          base_stride[i] != select_stride[i]) {
        dim = i;
        break;
      }
    }

    // Obtain start index and select length
    int64_t int_index = src_desc.offset_ / base_stride[dim];
    for (const auto i : c10::irange(base_size.size())) {
      if (i == dim) {
        start.emplace_back(int_index);
        length.emplace_back(1);
      } else {
        start.emplace_back(0);
        length.emplace_back(base_size[i]);
      }
    }
    return true;
  }

  void select_to_contiguous(at::Tensor &self, const at::Tensor &src,
                            c10::SmallVector<int64_t, MAX_DIM> &start,
                            c10::SmallVector<int64_t, MAX_DIM> &length,
                            const ContiguousTensorDesc &src_desc) {
    const auto &base_size = src_desc.base_sizes_;
    // Recover base tensor(necessary) a = b.select(1, 1)
    at::Tensor temp_src = TransContiguous::view_tensor(src, src_desc.base_offset_, base_size, src_desc.base_strides_);

    // construct StridedSlice param
    int64_t axis_size = static_cast<int64_t>(start.size());
    c10::SmallVector<int64_t, MAX_DIM> strides(axis_size, 1);
    c10::SmallVector<int64_t, MAX_DIM> end;
    int64_t shrink_mask = 0;
    for (int64_t i = 0; i < axis_size; ++i) {
      end.emplace_back(start[i] + length[i]);
      if (length[i] == 1 && temp_src.size(i) != 1) {
        shrink_mask += std::pow(2, i);
      }
    }

    // call StridedSlice op to contiguous
    custom_ops::npu_indexing_out(temp_src, start, end, strides, 0, 0, 0, 0, shrink_mask, self);
    return;
  }
}; // class SelectContiguousOpt

REGISTER_COPY_OPT(select, SelectContiguousOpt)

} // namespace native
} // namespace at_npu