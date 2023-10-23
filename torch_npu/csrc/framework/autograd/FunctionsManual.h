#pragma once

// NB: Must be at the top of file to avoid including the deprecated "math.h".
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#endif

#include <ATen/ATen.h>

#include "torch_npu/csrc/aten/Functions.h"

namespace at_npu {
namespace autograd {
namespace generated {
namespace details {

// A simple way to imperatively compute index ranges for slots
// that have been flattened
struct IndexRangeGenerator {
  IndexRange range(size_t range_size) {
    i += range_size;
    return {i - range_size, i};
  }
  size_t size() { return i; }
  private:
    size_t i = 0;
};

Tensor toNonOptFwGrad(const c10::optional<Tensor>& t);
Tensor toNonOptPrimal(const c10::optional<Tensor>& t);
Tensor toNonOptTensor(const c10::optional<Tensor>& t);

Tensor apply_loss_reduction(const Tensor& unreduced, int64_t reduction);
bool any_variable_defined(const variable_list& variables);
void copy_range(variable_list& out, IndexRange range, const at::Tensor& t);
void copy_range(variable_list& out, IndexRange range, at::ArrayRef<at::Tensor> t);
at::Tensor not_implemented(const char* name, const char* reason = "");
std::vector<Tensor> not_implemented_list(const char* name, const char* reason = "");

TORCH_API at::Tensor maybe_multiply(const at::Tensor& t, const at::Scalar& s);

} // namespace details
} // namespace generated
} // namespace autograd
} // namespace at_npu
