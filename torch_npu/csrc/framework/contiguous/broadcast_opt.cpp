#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace at_npu {
namespace native {

class BroadcastContiguousOpt : public ContiguousOpt {
public:
  bool Optimizer(at::Tensor &self, const at::Tensor &src,
                 const ContiguousTensorDesc &src_desc) override {
    if (self.dim() != src.dim()) {
      return false;
    }

    if (can_use_broadcast(src_desc)) {
      RECORD_FUNCTION("contiguous_d_BroadcastTo", std::vector<c10::IValue>({src}));
      bool can_contiguous = broadcast_to_contiguous(self, src, src_desc);
      return can_contiguous;
    }
    return false;
  }

private:
  bool can_use_broadcast(const ContiguousTensorDesc &src_desc) {
    // Reshape is used to process dimension addition cases for expand/expand_as.
    // Here, dimension expansion cases of expand/expand_as are processed.
    const auto &base_sizes = src_desc.base_sizes_;
    const auto &base_strides = src_desc.base_strides_;
    const auto &view_sizes = src_desc.sizes_;
    const auto &view_strides = src_desc.strides_;

    // The new ones will be appended at the front.
    // Any dimension of size 1 can be expanded to an arbitrary value.
    int64_t base_dim = static_cast<int64_t>(base_sizes.size());
    int64_t view_dim = static_cast<int64_t>(view_sizes.size());
    auto expand_dims = view_dim - base_dim;
    if (expand_dims < 0) {
      return false;
    }

    bool has_zero_in_stride = false;
    for (int64_t i = 0; i < base_dim; i++) {
      if (view_strides[i + expand_dims] == 0) {
        has_zero_in_stride = true;
        if (base_sizes[i] != 1 || view_sizes[i + expand_dims] == 1) {
          return false;
        }
      } else {
        if (view_sizes[i + expand_dims] != base_sizes[i] ||
            view_strides[i + expand_dims] != base_strides[i]) {
          return false;
        }
      }
    }

    for (auto i = 0; i < expand_dims; i++) {
      if (view_sizes[i] != 1 && view_strides[i] != 0) {
        return false;
      }
      has_zero_in_stride = true;
    }
    return has_zero_in_stride;
  }

  bool broadcast_to_contiguous(at::Tensor &self, const at::Tensor &src,
                               const ContiguousTensorDesc &src_desc) {
    std::vector<int64_t> src_size(src.dim());
    for (const auto i : c10::irange(src_desc.sizes_.size())) {
      if (src_desc.strides_[i] == 0) {
        src_size[i] = 1;
      } else {
        src_size[i] = src_desc.sizes_[i];
      }
    }

    // create contiguous tensor for npu BroadcastToD
    at::Tensor temp_src = at::empty({0}, src.options());
    temp_src.set_(src);
    temp_src.unsafeGetTensorImpl()->set_sizes_and_strides(src_size,
                                                          src.strides());

    if (temp_src.is_contiguous()) {
      custom_ops::npu_broadcast_out(temp_src, self.sizes(), self);
      return true;
    }
    return false;
  }
}; // class BroadcastContiguousOpt

REGISTER_COPY_OPT(broadcast, BroadcastContiguousOpt)

} // namespace native
} // namespace at_npu