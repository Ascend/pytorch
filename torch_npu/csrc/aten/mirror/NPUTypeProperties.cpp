#include "torch_npu/csrc/aten/mirror/NPUTypeProperties.h"

namespace at_npu {
namespace native {

static inline at::ScalarType promote_skip_undefined(at::ScalarType a, at::ScalarType b)
{
    if (a == at::ScalarType::Undefined) {
        return b;
    }
    if (b == at::ScalarType::Undefined) {
        return a;
    }
    return promoteTypes(a, b);
}

static inline at::ScalarType combine_categories(at::ScalarType higher, at::ScalarType lower)
{
    if (isFloatingType(higher)) {
        return higher;
    }
    if (higher == at::ScalarType::Bool || isFloatingType(lower)) {
        return promote_skip_undefined(higher, lower);
    }
    if (higher != at::ScalarType::Undefined) {
        return higher;
    }
    return lower;
}

ResultTypeState update_result_type_state(const at::Tensor &tensor, const ResultTypeState &in_state)
{
    if (!tensor.defined()) {
        return in_state;
    }
    ResultTypeState new_state = in_state;
    at::ScalarType current = tensor.scalar_type();
    if (tensor.unsafeGetTensorImpl()->is_wrapped_number() && isFloatingType(current)) {
        current = c10::typeMetaToScalarType(at::get_default_dtype());
    }
    if (tensor.dim() > 0) {
        new_state.dimResult = promote_skip_undefined(in_state.dimResult, current);
    } else if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
    } else {
        new_state.zeroResult = promote_skip_undefined(in_state.zeroResult, current);
    }

    return new_state;
}

at::ScalarType result_type(const ResultTypeState &in_state)
{
    return combine_categories(in_state.dimResult, combine_categories(in_state.zeroResult, in_state.wrappedResult));
}

at::ScalarType result_type(at::ScalarType a, at::ScalarType b)
{
    return promote_skip_undefined(a, b);
}

}
}
