#ifndef __PLUGIN_NATIVE_UTILS_NPU_TYPE_PROPERIES__
#define __PLUGIN_NATIVE_UTILS_NPU_TYPE_PROPERIES__


#include <ATen/ATen.h>

namespace at_npu {
namespace native {

struct ResultTypeState {
    at::ScalarType dimResult = at::ScalarType::Undefined;
    at::ScalarType wrappedResult = at::ScalarType::Undefined;
    at::ScalarType zeroResult = at::ScalarType::Undefined;
};

ResultTypeState update_result_type_state(const at::Tensor& tensor, const ResultTypeState& in_state);
at::ScalarType result_type(const ResultTypeState& state);
at::ScalarType result_type(at::ScalarType a, at::ScalarType b);

}
}

#endif // __NATIVE_NPU_UTILS_NPU_TYPE_PROPERIES__
