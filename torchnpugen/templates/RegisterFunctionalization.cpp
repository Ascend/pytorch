// ${generated_comment}

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>

#include <ATen/Operators.h>
#include <ATen/NativeFunctions.h>
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUException.h"


namespace at_npu {
namespace functionalization {

// This keyset is used by functionalization when it calls into meta kernels
// to accurately propagate stride metadata.
// Exclude any modes: the purpose of calling into meta kernels is only as an implementation
// detail to perform shape inference, and we don't want any modal keys to run.
// Specifically, we want to prevent functionalization and Python modes from running.
constexpr auto exclude_keys_for_meta_dispatch =
    c10::functorch_transforms_ks |
    c10::DispatchKeySet({
        c10::DispatchKey::FuncTorchDynamicLayerBackMode,
        c10::DispatchKey::FuncTorchDynamicLayerFrontMode,
        c10::DispatchKey::Python
    });

// Helper around at::has_internal_overlap.
// The ATen util is used in hot-path eager mode: it's always fast,
// but might return TOO_HARD sometimes.
// During functionalization, we're ok taking a bit longer
// to detect memory overlap.
inline bool has_internal_overlap_helper(const at::Tensor t)
{
    auto has_overlap = at::has_internal_overlap(t);
    if (has_overlap == at::MemOverlap::Yes) {
        return true;
    }
    if (has_overlap == at::MemOverlap::No) {
        return false;
    }
    return false;
}


inline at::Tensor to_meta(const at::Tensor& t)
{
    if (!t.defined()) return t;
    return at::native::empty_strided_meta_symint(t.sym_sizes(), t.sym_strides(),
        c10::make_optional(t.scalar_type()) /* dtype */, c10::make_optional(t.layout() /* layout */),
        c10::make_optional(c10::Device(at::kMeta)) /* device */, c10::nullopt /* pin_memory */);
}

inline c10::optional<at::Tensor> to_meta(const c10::optional<at::Tensor>& t)
{
    if (t.has_value()) {
        return c10::make_optional<at::Tensor>(to_meta(*t));
    }
    return c10::nullopt;
}

inline std::vector<at::Tensor> to_meta(at::ITensorListRef t_list)
{
    std::vector<at::Tensor> outputs;
    outputs.reserve(t_list.size());
    for (const auto& tensor : t_list) {
        outputs.push_back(to_meta(tensor));
    }
    return outputs;
}

inline c10::List<at::Tensor> to_meta(const c10::List<at::Tensor>& t_list)
{
    c10::List<at::Tensor> outputs;
    outputs.reserve(t_list.size());
    for (const auto i : c10::irange(t_list.size())) {
        outputs.push_back(to_meta(t_list[i]));
    }
    return outputs;
}

inline c10::List<c10::optional<at::Tensor>> to_meta(const c10::List<c10::optional<at::Tensor>>& t_list)
{
    c10::List<c10::optional<at::Tensor>> outputs;
    outputs.reserve(t_list.size());
    for (const auto i : c10::irange(t_list.size())) {
        outputs.push_back(to_meta(t_list[i]));
    }
    return outputs;
}


${func_definitions}

}  // namespace functionalization

namespace {

TORCH_LIBRARY_IMPL(npu, Functionalize, m) {
    ${func_registrations};
}

}  // namespace

} // namespace at_npu
