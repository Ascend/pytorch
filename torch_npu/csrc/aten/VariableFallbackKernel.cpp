#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUException.h"

/*
 * This file implements a variable fallback kernel for custom operators.
 * Since tensors always have the Autograd set, but custom operators
 * usually don't have a kernel registered for Autograd, the dispatcher
 * will call into this fallback kernel instead.
 * Note that this is not a correct autograd implementation. It will just
 * fallthrough to the custom operator implementation.
 * If you want a custom operator to work with autograd, you need to use
 * autograd::Function so that the custom operator implementation knows how to
 * do autograd.
 * Note also that ops from native_functions.yaml register their own variable
 * kernels, so this is never called for them.
 */

using c10::OperatorHandle;
using c10::Stack;
using c10::DispatchKey;
using c10::DispatchKeySet;
using c10::Dispatcher;
using c10::KernelFunction;

namespace {

// Register fallthrough for Autograd backends dispatch keys
// NB: But not the private use ones; maybe the extension wants
// to override it themselves!

// (Ascend) TORCH_LIBRARY_IMPL
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}

bool has_op_name_warned(const std::string& op_name)
{
    static std::unordered_set<std::string> _op_lists = {};
    if (_op_lists.find(op_name) != _op_lists.end()) {
        return true;
    }
    _op_lists.insert(op_name);
    return false;
}

void npu_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
    if (!has_op_name_warned(c10::toString(op.schema().operator_name()))) {
        TORCH_NPU_WARN("CAUTION: The operator '",
                       op.schema().operator_name(),
                       "' is not currently supported ",
                       "on the NPU backend and will fall back to run on the CPU.",
                       " This may have performance implications.");
    }
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&npu_cpu_fallback>());
}

void npu_Sparse_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
    TORCH_CHECK(false, "CAUTION: The operator '",
                op.schema().operator_name(),
                "' is not currently supported on the NPU backend.", OPS_ERROR(ErrCode::NOT_SUPPORT))
}

TORCH_LIBRARY_IMPL(_, SparsePrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&npu_Sparse_fallback>());
}
}
