#ifndef __PULGIN_NATIVE_CONTIGUOUS_CONTIGUOUS_OPTIMIZE__
#define __PULGIN_NATIVE_CONTIGUOUS_CONTIGUOUS_OPTIMIZE__

#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/contiguous/contiguous_register.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include <ATen/record_function.h>

namespace at_npu {
namespace native {

    constexpr int MaxCombinedCasesNum = 2;
    constexpr int ViewAndBaseInfoStackNum = 2;
    using ShapeStrideStack =
            c10::SmallVector<c10::SmallVector<FormatShape, ViewAndBaseInfoStackNum>,
                    MaxCombinedCasesNum>;
    using OffsetStack = c10::SmallVector<int64_t, MaxCombinedCasesNum>;

    constexpr int CachedMaxSize = 1000000;
    constexpr int CachedOptParaNum = 5;
    struct CachedContiguousOpt {
        string cached_opt_case;
        c10::SmallVector<c10::SmallVector<int64_t, MAX_DIM>, CachedOptParaNum> cached_opt_parameters;
        ShapeStrideStack shape_stride_stack;
        OffsetStack offset_stack;
        ContiguousTensorDesc contiguous_tensor_desc;
    };

class TransContiguous {
public:
    TransContiguous() {}
    virtual ~TransContiguous() {}
    static bool CheckClone(const at::Tensor &src, at::Tensor &self);
    static ContiguousTensorDesc
    GetTensorDescInfo(const at::Tensor &src,
                    const OptimizationCases &opt_cases = optCasesDefault);
    static bool can_optimize_(ContiguousTensorDesc &tensor_desc);
    static bool CanOptimize(ContiguousTensorDesc &tensor_desc);
    static bool CanOptimize(const at::Tensor &tensor,
                          const OptimizationCases &opt_cases);
    static bool
    contiguous_optimize_with_anyformat_(at::Tensor &self, const at::Tensor &src,
                                      ContiguousTensorDesc &src_desc);
    static bool ContiguousOptimizeWithAnyFormat(
        at::Tensor &self, const at::Tensor &src,
        const OptimizationCases &opt_cases = optCasesAnyFormat);
    static c10::optional<at::Tensor> ContiguousOptimizeWithAnyFormat(
        const at::Tensor &src,
        const OptimizationCases &opt_cases = optCasesAnyFormat);
    static bool ContiguousOptimizeWithBaseFormat(
        at::Tensor &self, const at::Tensor &src,
        const OptimizationCases &opt_cases = optCasesDefault,
        bool OpenCombined = true);
    static bool cached_contiguous_optimize_with_anyformat_(
            at::Tensor &self, const at::Tensor &src, ContiguousTensorDesc &src_desc);
    static ska::flat_hash_map<size_t, CachedContiguousOpt> cached_contiguous_opt;
    static at::Tensor view_tensor(const at::Tensor& self, int64_t offset, const c10::IntArrayRef& sizes, const c10::IntArrayRef& strides);

private:
    static OptimizationCases optCasesDefault;
    static OptimizationCases optCasesAnyFormat;
};

} // namespace native
} // namespace at_npu

#endif