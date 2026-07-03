
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


}
}
