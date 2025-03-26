#include "torch_npu/csrc/framework/contiguous/ContiguousOpt.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include <ATen/quantized/QTensorImpl.h>

namespace at_npu {
namespace native {

OptimizationCases TransContiguous::optCasesDefault = {};
OptimizationCases TransContiguous::optCasesAnyFormat = {"reshape", "slice"};
ska::flat_hash_map<size_t, CachedContiguousOpt> TransContiguous::cached_contiguous_opt;


ContiguousTensorDesc TransContiguous::GetTensorDescInfo(
    const at::Tensor &src, const OptimizationCases &opt_cases)
{
    auto src_base_info = torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc();
    c10::SmallVector<int64_t, MAX_DIM> src_size_inferred;
    c10::SmallVector<int64_t, MAX_DIM> src_stride_inferred;
    c10::SmallVector<int64_t, MAX_DIM> src_storage_size_inferred = src_base_info.storage_sizes_;
    if (src.dim() == 0) {
        src_size_inferred = {1};
        src_stride_inferred = {1};
        if (src_storage_size_inferred.size() == 0) {
            src_storage_size_inferred = {1};
        }
    } else {
        src_size_inferred = CalcuOpUtil::ConvertIntArrayRefToSmallVector(src.sizes());
        src_stride_inferred = CalcuOpUtil::ConvertIntArrayRefToSmallVector(src.strides());
    }
    ContiguousTensorDesc src_desc = {
        src.is_contiguous(),       src_size_inferred,
        src_stride_inferred,       src.storage_offset(),
        src_base_info.base_sizes_, src_base_info.base_strides_,
        src_storage_size_inferred, src_base_info.base_offset_,
        src_base_info.npu_format_, opt_cases};
    if (src_desc.opt_cases_.empty()) {
        src_desc.find_match_optimization_cases();
    }
    return src_desc;
}

bool TransContiguous::CheckClone(const at::Tensor &src, at::Tensor &self)
{
    // self tensor may not be temporary constructed empty tensor from src, so:
    // 1. contiguous storage is needed:storage_offset and numels eq
    // 2. full memory copy: size match between src and self
    if (StorageDescHelper::OffsetAreMatch(&self) && self.is_contiguous() &&
        src.sizes().equals(self.sizes()) &&
        self.sizes().equals(torch_npu::NPUBridge::GetNpuStorageImpl(self)->get_npu_desc().base_sizes_)) {
        return true;
    }
    return false;
}

bool TransContiguous::can_optimize_(ContiguousTensorDesc &tensor_desc)
{
    for (auto opt_case : tensor_desc.opt_cases_) {
        bool res = register_opt::CopyOptRegister::GetInstance()->CanOptimize(
            opt_case, tensor_desc);
        if (res) {
            // refresh patterns to only keep optimized pattern
            tensor_desc.opt_cases_.clear();
            tensor_desc.opt_cases_.emplace_back(opt_case);
            return true;
        }
    }
    return false;
}

bool TransContiguous::CanOptimize(ContiguousTensorDesc &tensor_desc)
{
    return can_optimize_(tensor_desc);
}

bool TransContiguous::CanOptimize(const at::Tensor &tensor,
                                  const OptimizationCases &opt_cases)
{
    ContiguousTensorDesc tensor_desc = GetTensorDescInfo(tensor, opt_cases);
    return can_optimize_(tensor_desc);
}

bool TransContiguous::contiguous_optimize_with_anyformat_(
    at::Tensor &self, const at::Tensor &src, ContiguousTensorDesc &src_desc)
{
    if (!CheckClone(src, self)) {
        return false;
    }
    for (auto &opt_case : src_desc.opt_cases_) {
        bool res = register_opt::CopyOptRegister::GetInstance()->Run(opt_case, self,
                                                                     src, src_desc);
        if (res) {
            return true;
        }
    }
    return false;
}

size_t GetHash_(const c10::SmallVector<int64_t, MAX_DIM>& small_vector_size)
{
    size_t seed = 0;
    for (size_t i = 0; i < small_vector_size.size(); i++) {
        seed ^= static_cast<size_t>(small_vector_size[i]) + (seed << 6) + (seed >> 2);
    }
    return seed;
}

size_t GetHash_(const ContiguousTensorDesc &src_desc)
{
    size_t hash_src_desc = (GetHash_(src_desc.sizes_)<<52) +
                           (GetHash_(src_desc.base_sizes_)<<40) +
                           (GetHash_(src_desc.strides_)<<28) +
                           (GetHash_(src_desc.base_strides_)<<16) +
                           (static_cast<size_t>(src_desc.offset_) << 4) +
                           src_desc.npu_format_;
    return hash_src_desc;
}

bool equalDesc(const ContiguousTensorDesc &src_desc, const ContiguousTensorDesc &desc_desc)
{
    if (src_desc.sizes_ == desc_desc.sizes_ &&
        src_desc.base_sizes_ == desc_desc.base_sizes_ &&
        src_desc.strides_ == desc_desc.strides_ &&
        src_desc.base_strides_ == desc_desc.base_strides_ &&
        src_desc.offset_ == desc_desc.offset_ &&
        src_desc.npu_format_ == desc_desc.npu_format_) {
        return true;
    }
    return false;
}

bool TransContiguous::cached_contiguous_optimize_with_anyformat_(
    at::Tensor &self, const at::Tensor &src, ContiguousTensorDesc &src_desc)
{
    // No cached, try caching
    if (!CheckClone(src, self)) {
        return false;
    }
    src_desc.hash_src_desc = GetHash_(src_desc);
    auto it = TransContiguous::cached_contiguous_opt.find(src_desc.hash_src_desc);
    if (it != TransContiguous::cached_contiguous_opt.end()) {
        // Cached
        if (equalDesc(src_desc, it->second.contiguous_tensor_desc)) {
            src_desc.cached_contiguous = true;
            auto &opt_case = it->second.cached_opt_case;
            return register_opt::CopyOptRegister::GetInstance()->CachedRun(opt_case, self,
                                                                           src, src_desc);
        }
        return contiguous_optimize_with_anyformat_(self, src, src_desc);
    }

    for (auto &opt_case : src_desc.opt_cases_) {
        bool res = false;
        if (TransContiguous::cached_contiguous_opt.size() >= CachedMaxSize) {
            res = register_opt::CopyOptRegister::GetInstance()->Run(opt_case, self, src, src_desc);
        } else {
            src_desc.cached_contiguous = false;
            res =  register_opt::CopyOptRegister::GetInstance()->CachedRun(opt_case, self, src, src_desc);
        }
        if (res) {
            return true;
        }
    }
    return false;
}

bool TransContiguous::ContiguousOptimizeWithAnyFormat(
    at::Tensor &self, const at::Tensor &src,
    const OptimizationCases &opt_cases)
{
    ContiguousTensorDesc src_desc = GetTensorDescInfo(src, opt_cases);
    return contiguous_optimize_with_anyformat_(self, src, src_desc);
}

c10::optional<at::Tensor> TransContiguous::ContiguousOptimizeWithAnyFormat(
    const at::Tensor &src, const OptimizationCases &opt_cases)
{
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
        "Expected all tensors to be on the same device. "
        "Expected NPU tensor, please check whether the input tensor device is correct.",
        OPS_ERROR(ErrCode::TYPE));
    auto self = OpPreparation::ApplyTensorWithFormat(
        src.sizes(), src.options(), torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc().npu_format_);
    ContiguousTensorDesc src_desc = GetTensorDescInfo(src, opt_cases);
    if (cached_contiguous_optimize_with_anyformat_(self, src, src_desc)) {
        return self;
    }
    return c10::nullopt;
}

bool TransContiguous::ContiguousOptimizeWithBaseFormat(
    at::Tensor &self, const at::Tensor &src, const OptimizationCases &opt_cases,
    bool OpenCombined)
{
    TORCH_CHECK(FormatHelper::IsBaseFormatType(src),
                "ContiguousOptimizeWithBaseFormat func requires Input Tensor "
                "with base format!", OPS_ERROR(ErrCode::TYPE));
    // In non-specific cases, classify the cases and simplify judgement.
    ContiguousTensorDesc src_desc = GetTensorDescInfo(src, opt_cases);
    if (OpenCombined &&
        c10_npu::option::OptionsManager::CheckCombinedOptimizerEnable()) {
        src_desc.add_optimization_case("combined");
    }
    return cached_contiguous_optimize_with_anyformat_(self, src, src_desc);
}


at::Tensor TransContiguous::view_tensor(const at::Tensor& self,
                                        int64_t offset,
                                        const c10::IntArrayRef& sizes,
                                        const c10::IntArrayRef& strides)
{
    at::Tensor self_;
    if (self.is_quantized()) {
        self_ = at::detail::make_tensor<at::QTensorImpl>(
                c10::TensorImpl::VIEW,
                c10::Storage(self.storage()),
                self.key_set(),
                self.dtype(),
                get_qtensorimpl(self)->quantizer());
    } else {
        self_ = at::detail::make_tensor<at::TensorImpl>(
                c10::TensorImpl::VIEW,
                c10::Storage(self.storage()),
                self.key_set(),
                self.dtype());
    }
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset(offset);
    self_tmp_->set_sizes_and_strides(sizes, strides);
    at::namedinference::propagate_names(self_, self);
    return self_;
}

} // namespace native
} // namespace at_npu