#include <ATen/ATen.h>

namespace at_npu {
namespace native {

at::Tensor isnan_npu(const at::Tensor& self)
{
    return self != self;
}

bool is_nonzero_npu(const at::Tensor& self)
{
    c10::Scalar localScalar = self.item();
    if (localScalar.isFloatingPoint()) {
        return localScalar.to<double>() != 0;
    } else if (localScalar.isIntegral(false)) {
        return localScalar.to<int64_t>() != 0;
    } else if (localScalar.isBoolean()) {
        return localScalar.to<bool>();
    }

    return false;
}

} // namespace native
} // namespace at_npu
