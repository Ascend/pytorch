#ifndef __PULGIN_NATIVE_NPU_UTILS_ADVANCED_INDEX__
#define __PULGIN_NATIVE_NPU_UTILS_ADVANCED_INDEX__

#include <ATen/native/IndexingUtils.h>
#include <ATen/ExpandUtils.h>

namespace at_npu {
namespace native {

struct AdvancedIndex {
    AdvancedIndex(const at::Tensor& src, at::TensorList indices);
    at::Tensor src;
    std::vector<at::Tensor> indices;
    at::DimVector indexed_sizes;
    at::DimVector indexed_strides;
    int64_t dims_before;
    int64_t dims_after;
};

class AdvanceIndex {
public:
  static bool all_strides_match(at::TensorList tensors);
  static at::Tensor reshape_indexer(const at::Tensor& index, int64_t dims_before, int64_t dims_after);
  static at::Tensor restride_src(const at::Tensor& src, int64_t dims_before, int64_t dims_indexed,
      at::IntArrayRef replacement_shape);
  static std::string shapes_as_str(at::TensorList tensors);
  static AdvancedIndex make_info(at::Tensor self, const torch::List<c10::optional<at::Tensor>>& orig);
  static std::vector<at::Tensor> npu_expand_tensors(
      const at::Tensor& self,
      const torch::List<c10::optional<at::Tensor>>& indices);
};

} // namespace native
} // namespace at_npu

#endif
