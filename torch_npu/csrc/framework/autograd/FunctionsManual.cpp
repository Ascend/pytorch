#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "FunctionsManual.h"

// Helper functions for autogenerated code
// These used to be inlined into the codegened Functions.cpp

namespace at_npu {
namespace autograd {
namespace generated {
namespace details {

using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;
using at::areAnyTensorSubclassLike;

Tensor apply_loss_reduction(const Tensor& unreduced, int64_t reduction)
{
    if (reduction == at::Reduction::Mean) {
        return unreduced.mean();
    } else if (reduction == at::Reduction::Sum) {
        return unreduced.sum();
    }
    return unreduced;
}

bool any_variable_defined(const variable_list& variables)
{
    for (const auto& variable : variables) {
        if (variable.defined()) {
            return true;
        }
    }
    return false;
}

bool isDefined(const c10::optional<Tensor>& t)
{
    return t.has_value() && t->defined();
}

Tensor toNonOptTensor(const c10::optional<Tensor>& t)
{
    return t.has_value() ? *t : Tensor();
}

Tensor toNonOptFwGrad(const c10::optional<Tensor>& t)
{
    // 0: level 0
    return (t.has_value() && t->defined()) ? t->_fw_grad(0) : Tensor();
}

Tensor toNonOptPrimal(const c10::optional<Tensor>& t) {
    // 0: level 0
    return (t.has_value() && t->defined()) ? t->_fw_primal(0) : Tensor();
}

void copy_range(variable_list& out, IndexRange range, const Tensor& t)
{
    AT_ASSERT(range.second <= out.size(), OPS_ERROR(ErrCode::PARAM));
    AT_ASSERTM(range.second - range.first == 1,
               "inconsistent range for Tensor output",
               OPS_ERROR(ErrCode::PARAM));
    out[range.first] = t;
}

void copy_range(variable_list& out, IndexRange range, at::ArrayRef<Tensor> t)
{
    AT_ASSERT(range.second <= out.size(), OPS_ERROR(ErrCode::PARAM));
    AT_ASSERTM(range.second - range.first == t.size(),
               "inconsistent range for TensorList output",
               OPS_ERROR(ErrCode::PARAM));
    std::copy(t.begin(), t.end(), out.begin() + range.first);
}

template <typename T>
T not_implemented_base(const char* name, const char* reason)
{
    std::string msg = c10::str("the derivative for '", name, "' is not implemented.");
    if (strlen(reason) > 0) {
        msg = c10::str(msg, " ", reason);
    };
    TORCH_CHECK_NOT_IMPLEMENTED(false, msg);
}

Tensor not_implemented(const char* name, const char* reason)
{
    return not_implemented_base<Tensor>(name, reason);
}

std::vector<Tensor> not_implemented_list(const char* name, const char* reason)
{
    return not_implemented_base<std::vector<Tensor>>(name, reason);
}

Tensor maybe_multiply(const Tensor& t, const Scalar& s)
{
    bool is_one = false;
    if (s.isFloatingPoint()) {
        is_one = s.toSymFloat() == 1;
    } else if (s.isIntegral(true)) {
        is_one = s.toSymInt() == 1;
    }

    if (is_one) {
        return t;
    } else {
        return t * s;
    }
}

} // namespace details
} // namespace generated
} // namespace autograd
} // namespace at_npu
