#pragma once

#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/autograd/function.h>

#include <torch_npu/csrc/_compat/version.h>

#include <memory>
#include <utility>

// Compatibility layer for autograd Node smart-pointer changes introduced in
// PyTorch 2.13 (#181782): grad_fn allocation was migrated from
//   std::shared_ptr<Op>(new Op(...), torch::autograd::deleteNode)
// to
//   c10::make_intrusive<Op>(...)
// and SavedVariable::unpack() takes c10::intrusive_ptr<Node> instead of
// std::shared_ptr<Node>. deleteNode was removed entirely.
//
// CAN REMOVE the version branches below when MIN_SUPPORTED >= (2, 13).

namespace torch_npu {
namespace compat {

#if TORCH_NPU_VERSION_GE(2, 13)

template <typename T>
using GradFnPtr = c10::intrusive_ptr<T>;

template <typename Op, typename... Args>
inline GradFnPtr<Op> make_grad_fn(Args&&... args)
{
    return c10::make_intrusive<Op>(std::forward<Args>(args)...);
}

#else

template <typename T>
using GradFnPtr = std::shared_ptr<T>;

template <typename Op, typename... Args>
inline GradFnPtr<Op> make_grad_fn(Args&&... args)
{
    return std::shared_ptr<Op>(
        new Op(std::forward<Args>(args)...), torch::autograd::deleteNode);
}

#endif

// Type used by SavedVariable::unpack() — see comment at top of file.
using SavedForPtr = GradFnPtr<torch::autograd::Node>;

}  // namespace compat
}  // namespace torch_npu
