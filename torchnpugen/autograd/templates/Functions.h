#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/TensorGeometry.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/Export.h>

#include <c10/core/SymIntArrayRef.h>

// COMPAT(>= 2.11): SavedVariable::unpack parameter type changed from std::shared_ptr<Node>
// to c10::intrusive_ptr<Node> in PyTorch 2.11
// We detect the change by checking if TORCH_VERSION_MAJOR >= 2 and TORCH_VERSION_MINOR >= 11
#if defined(TORCH_VERSION_MAJOR) && defined(TORCH_VERSION_MINOR)
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 11
#define USE_INTRUSIVE_PTR_FOR_UNPACK 1
#else
#define USE_INTRUSIVE_PTR_FOR_UNPACK 0
#endif
#else
// Fallback: assume new API if version macros not defined (likely newer PyTorch)
#define USE_INTRUSIVE_PTR_FOR_UNPACK 1
#endif

using namespace torch::autograd;

namespace at_npu { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::ArrayRef;
using at::Type;
using at::TensorGeometry;
using at::ScalarType;
using c10::optional;
using c10::fmap;

#if USE_INTRUSIVE_PTR_FOR_UNPACK
inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs, c10::intrusive_ptr<Node> saved_for = nullptr)
{
    // NB: we must explicitly do the conversion in the lambda, otherwise template
    // deduction will give a Tensor of Variable which is not convertible
    return fmap(xs, [&saved_for](const SavedVariable& x) {
        return static_cast<Tensor>(x.unpack(saved_for));
    });
}

inline c10::List<c10::optional<Tensor>> unpack_opt_list(at::ArrayRef<SavedVariable> xs, c10::intrusive_ptr<Node> saved_for = nullptr)
{
    torch::List<c10::optional<Tensor>> result;
    result.reserve(xs.size());
    for (const SavedVariable& v : xs) {
        auto var = v.unpack(saved_for);
        result.push_back(var.defined() ? c10::optional<Tensor>(var) : c10::nullopt);
    }
    return result;
}
#else
inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs, std::shared_ptr<Node> saved_for = nullptr)
{
    // NB: we must explicitly do the conversion in the lambda, otherwise template
    // deduction will give a Tensor of Variable which is not convertible
    return fmap(xs, [&saved_for](const SavedVariable& x) {
        return static_cast<Tensor>(x.unpack(saved_for));
    });
}

inline c10::List<c10::optional<Tensor>> unpack_opt_list(at::ArrayRef<SavedVariable> xs, std::shared_ptr<Node> saved_for = nullptr)
{
    torch::List<c10::optional<Tensor>> result;
    result.reserve(xs.size());
    for (const SavedVariable& v : xs) {
        auto var = v.unpack(saved_for);
        result.push_back(var.defined() ? c10::optional<Tensor>(var) : c10::nullopt);
    }
    return result;
}
#endif


struct TypeAndSize {
    TypeAndSize() : options(at::TensorOptions()) {}
    /* implicit */
    TypeAndSize(const Tensor & t)
        : sizes(t.sizes().vec())
        , options(t.options()) {}

    Tensor zeros() { return at::zeros(sizes, options); }

private:
    std::vector<int64_t> sizes;
    at::TensorOptions options;
};

${autograd_function_declarations}

}}} // namespace at_npu::autograd::generated
