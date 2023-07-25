#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

#include "torch_npu/csrc/aten/aten_foreach_ops.h"
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
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m){
  m.fallback(torch::CppFunction::makeFallthrough());
}

bool has_op_name_warned(const std::string& op_name) {
  static std::unordered_set<std::string> _op_lists = {};
  if (_op_lists.find(op_name) != _op_lists.end()) {
    return true;
  }
  _op_lists.insert(op_name);
  return false;
}

void npu_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
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

#define FOREACH_IMPL_WITHOUT_OVERLOAD(NAME)                                         \
  m.impl("_foreach_"#NAME, TORCH_FN(at::native::foreach_tensor_##NAME##_slow));     \
  m.impl("_foreach_"#NAME"_", TORCH_FN(at::native::foreach_tensor_##NAME##_slow_));

#define FOREACH_IMPL_WITH_SCALAR_LIST_SCALARLIST(NAME)                                                           \
  m.impl("_foreach_"#NAME".Scalar", TORCH_FN(at::native::foreach_tensor_##NAME##_scalar_kernel_slow));           \
  m.impl("_foreach_"#NAME"_.Scalar", TORCH_FN(at::native::foreach_tensor_##NAME##_scalar_kernel_slow_));         \
  m.impl("_foreach_"#NAME".List", TORCH_FN(at::native::foreach_tensor_##NAME##_list_kernel_slow));               \
  m.impl("_foreach_"#NAME"_.List", TORCH_FN(at::native::foreach_tensor_##NAME##_list_kernel_slow_));             \
  m.impl("_foreach_"#NAME".ScalarList", TORCH_FN(at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow));   \
  m.impl("_foreach_"#NAME"_.ScalarList", TORCH_FN(at::native::foreach_tensor_##NAME##_scalarlist_kernel_slow_));

#define FOREACH_IMPL_WITH_SCALAR_SCALARLIST_TENSOR(NAME)                                                   \
  m.impl("_foreach_"#NAME".Scalar", TORCH_FN(at::native::foreach_tensor_##NAME##_scalar_slow));            \
  m.impl("_foreach_"#NAME"_.Scalar", TORCH_FN(at::native::foreach_tensor_##NAME##_scalar_slow_));          \
  m.impl("_foreach_"#NAME".ScalarList", TORCH_FN(at::native::foreach_tensor_##NAME##_scalarlist_slow));    \
  m.impl("_foreach_"#NAME"_.ScalarList", TORCH_FN(at::native::foreach_tensor_##NAME##_scalarlist_slow_));  \
  m.impl("_foreach_"#NAME".Tensor", TORCH_FN(at::native::foreach_tensor_##NAME##_tensor_slow));            \
  m.impl("_foreach_"#NAME"_.Tensor", TORCH_FN(at::native::foreach_tensor_##NAME##_tensor_slow_));

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  FOREACH_IMPL_WITH_SCALAR_LIST_SCALARLIST(add)
  FOREACH_IMPL_WITH_SCALAR_LIST_SCALARLIST(sub)
  FOREACH_IMPL_WITH_SCALAR_LIST_SCALARLIST(mul)
  FOREACH_IMPL_WITH_SCALAR_LIST_SCALARLIST(div)
  FOREACH_IMPL_WITH_SCALAR_LIST_SCALARLIST(clamp_min)
  FOREACH_IMPL_WITH_SCALAR_LIST_SCALARLIST(clamp_max)
  FOREACH_IMPL_WITHOUT_OVERLOAD(abs)
  FOREACH_IMPL_WITHOUT_OVERLOAD(cos)
  FOREACH_IMPL_WITHOUT_OVERLOAD(acos)
  FOREACH_IMPL_WITHOUT_OVERLOAD(asin)
  FOREACH_IMPL_WITHOUT_OVERLOAD(atan)
  FOREACH_IMPL_WITHOUT_OVERLOAD(ceil)
  FOREACH_IMPL_WITHOUT_OVERLOAD(cosh)
  FOREACH_IMPL_WITHOUT_OVERLOAD(erf)
  FOREACH_IMPL_WITHOUT_OVERLOAD(erfc)
  FOREACH_IMPL_WITHOUT_OVERLOAD(exp)
  FOREACH_IMPL_WITHOUT_OVERLOAD(expm1)
  FOREACH_IMPL_WITHOUT_OVERLOAD(floor)
  FOREACH_IMPL_WITHOUT_OVERLOAD(frac)
  FOREACH_IMPL_WITHOUT_OVERLOAD(log)
  FOREACH_IMPL_WITHOUT_OVERLOAD(log10)
  FOREACH_IMPL_WITHOUT_OVERLOAD(log1p)
  FOREACH_IMPL_WITHOUT_OVERLOAD(log2)
  FOREACH_IMPL_WITHOUT_OVERLOAD(neg)
  FOREACH_IMPL_WITH_SCALAR_LIST_SCALARLIST(pow)
  FOREACH_IMPL_WITHOUT_OVERLOAD(reciprocal)
  FOREACH_IMPL_WITHOUT_OVERLOAD(round)
  FOREACH_IMPL_WITHOUT_OVERLOAD(sigmoid)
  FOREACH_IMPL_WITHOUT_OVERLOAD(sin)
  FOREACH_IMPL_WITHOUT_OVERLOAD(sinh)
  FOREACH_IMPL_WITHOUT_OVERLOAD(sqrt)
  FOREACH_IMPL_WITHOUT_OVERLOAD(tan)
  FOREACH_IMPL_WITHOUT_OVERLOAD(tanh)
  FOREACH_IMPL_WITHOUT_OVERLOAD(trunc)
  FOREACH_IMPL_WITHOUT_OVERLOAD(sqrt)
  FOREACH_IMPL_WITHOUT_OVERLOAD(tan)
  FOREACH_IMPL_WITHOUT_OVERLOAD(tanh)
  FOREACH_IMPL_WITHOUT_OVERLOAD(lgamma)
  FOREACH_IMPL_WITH_SCALAR_SCALARLIST_TENSOR(addcdiv)
  FOREACH_IMPL_WITH_SCALAR_SCALARLIST_TENSOR(addcmul)

  m.impl("_foreach_lerp.List", TORCH_FN(at::native::foreach_tensor_ternary_lerp_slow));
  m.impl("_foreach_lerp_.List", TORCH_FN(at::native::foreach_tensor_ternary_lerp_slow_));
  m.impl("_foreach_lerp.Scalar", TORCH_FN(at::native::foreach_tensor_lerp_list_kernel_slow));
  m.impl("_foreach_lerp_.Scalar", TORCH_FN(at::native::foreach_tensor_lerp_list_kernel_slow_));
  m.impl("_foreach_lerp.List", TORCH_FN(at::native::foreach_tensor_ternary_lerp_slow));

  // foreach_minimum/maximum dispatches to clamp_max/min
  m.impl("_foreach_maximum.Scalar", TORCH_FN(at::native::foreach_tensor_clamp_max_scalar_kernel_slow));
  m.impl("_foreach_maximum_.Scalar", TORCH_FN(at::native::foreach_tensor_clamp_max_scalar_kernel_slow_));         
  m.impl("_foreach_maximum.List", TORCH_FN(at::native::foreach_tensor_clamp_max_list_kernel_slow));         
  m.impl("_foreach_maximum_.List", TORCH_FN(at::native::foreach_tensor_clamp_max_list_kernel_slow_));             
  m.impl("_foreach_maximum.ScalarList", TORCH_FN(at::native::foreach_tensor_clamp_max_scalarlist_kernel_slow));
  m.impl("_foreach_maximum_.ScalarList", TORCH_FN(at::native::foreach_tensor_clamp_max_scalarlist_kernel_slow_));
  m.impl("_foreach_minimum.Scalar", TORCH_FN(at::native::foreach_tensor_clamp_min_scalar_kernel_slow)); 
  m.impl("_foreach_minimum_.Scalar", TORCH_FN(at::native::foreach_tensor_clamp_min_scalar_kernel_slow_));         
  m.impl("_foreach_minimum.List", TORCH_FN(at::native::foreach_tensor_clamp_min_list_kernel_slow));         
  m.impl("_foreach_minimum_.List", TORCH_FN(at::native::foreach_tensor_clamp_min_list_kernel_slow_));             
  m.impl("_foreach_minimum.ScalarList", TORCH_FN(at::native::foreach_tensor_clamp_min_scalarlist_kernel_slow));
  m.impl("_foreach_minimum_.ScalarList", TORCH_FN(at::native::foreach_tensor_clamp_min_scalarlist_kernel_slow_));
}

}
