#include "torch_npu/csrc/framework/contiguous/ReshapeOpt.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace at_npu {
namespace native {

class ReshapeV2ContiguousOpt : public ContiguousOpt {
public:
  bool Optimizer(at::Tensor &result, const at::Tensor &src,
                 const ContiguousTensorDesc &src_desc) override {
    ContiguousTensorDesc result_desc = TransContiguous::GetTensorDescInfo(result);
    if (check_reshape_match(result_desc, src_desc)) {
      if (can_use_memory_repoint(src_desc) &&
          reshape_match_by_memory_repoint(src, result)) {
        return true;
      }
      RECORD_FUNCTION("contiguous_d_Reshape", std::vector<c10::IValue>({src}));
      custom_ops::npu_reshape_out(src, src.sizes(), false, result);
      return true;
    }
    return false;
  }

  bool CanOptimizer(const ContiguousTensorDesc &src_desc) override {
    return check_reshape_match(src_desc);
  }

private:
  template <typename dataDtype>
  void ResetDataPtr(const at::Tensor &src, at::Tensor &self, dataDtype *value) {
    dataDtype *src_data_ptr = value + src.storage_offset();
    at::DataPtr self_data_ptr =
        at::DataPtr(src_data_ptr, self.storage().device());
    self.storage().set_data_ptr(std::move(self_data_ptr));
  }

  bool reshape_match_by_memory_repoint(const at::Tensor &src,
                                       at::Tensor &self) {
    RECORD_FUNCTION("contiguous_h_memRepoint", std::vector<c10::IValue>({src}));
    switch (src.scalar_type()) {
        case at::ScalarType::Half:
            ResetDataPtr(src, self,
                         static_cast<at::Half *>(src.storage().data_ptr().get()));
            return true;
        case at::ScalarType::BFloat16:
            ResetDataPtr(src, self,
                         static_cast<at::BFloat16 *>(src.storage().data_ptr().get()));
            return true;
        case at::ScalarType::Float:
            ResetDataPtr(src, self,
                         static_cast<float *>(src.storage().data_ptr().get()));
            return true;
        case at::ScalarType::Byte:
            ResetDataPtr(src, self,
                         static_cast<uint8_t *>(src.storage().data_ptr().get()));
            return true;
        case at::ScalarType::Char:
            ResetDataPtr(src, self,
                         static_cast<int8_t *>(src.storage().data_ptr().get()));
            return true;
        case at::ScalarType::Short:
            ResetDataPtr(src, self,
                         static_cast<int16_t *>(src.storage().data_ptr().get()));
            return true;
        case at::ScalarType::Int:
            ResetDataPtr(src, self,
                         static_cast<int *>(src.storage().data_ptr().get()));
            return true;
        case at::ScalarType::Long:
            ResetDataPtr(src, self,
                         static_cast<int64_t *>(src.storage().data_ptr().get()));
            return true;
        default:
            // Turn to conducting d2dCopyAsync for other dtypes.
            return false;
    }
  }

  bool can_use_memory_repoint(const ContiguousTensorDesc &src_desc) {
    if (FormatHelper::IsBaseFormatType(src_desc.npu_format_)) {
      return true;
    }

    if (src_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ) {
      // No padding
      if ((src_desc.sizes_[src_desc.sizes_.size() - 1] % 16 == 0) &&
          (src_desc.sizes_[src_desc.sizes_.size() - 2] % 16 == 0)) {
        return true;
      }
      return false;
    }
    return false;
  }
}; // class ReshapeV2ContiguousOpt

REGISTER_COPY_OPT(reshapeV2, ReshapeV2ContiguousOpt)

} // namespace native
} // namespace at_npu