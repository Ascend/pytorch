#include <ATen/NamedTensorUtils.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::full_out(at::IntArrayRef size, const at::Scalar& fill_value, at::Tensor& out)
{
    OpPreparation::CheckOut(
        {},
        out,
        out,
        size);
    out.fill_(fill_value);
    return out;
}

at::Tensor NPUNativeFunctions::full(
    at::IntArrayRef size,
    const at::Scalar& fill_value,
    c10::optional<at::DimnameList> names,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt)
{
    c10::TensorOptions option = c10::TensorOptions().dtype(dtype_opt)
                                                    .device(device_opt)
                                                    .layout(layout_opt)
                                                    .pinned_memory(pin_memory_opt);
    at::Tensor result = OpPreparation::ApplyTensorWithSizes(size, option);

    if (!dtype_opt.has_value()) {
        if (fill_value.isBoolean()) {
            option = option.dtype(at::kBool);
        } else if (fill_value.isIntegral(false)) {
            option = option.dtype(at::kLong);
        } else {
            option = option.dtype(c10::get_default_dtype());
        }
    }

    auto maybe_name = names.value_or(at::ArrayRef<at::Dimname>{});
    at::namedinference::propagate_names_if_nonempty(result, maybe_name);
    return result.fill_(fill_value);
}

}
}
