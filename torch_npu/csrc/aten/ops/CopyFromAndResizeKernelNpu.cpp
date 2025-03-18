#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst)
{
    TORCH_CHECK(dst.defined(), "dst is undefined", OPS_ERROR(ErrCode::NOT_SUPPORT));
    TORCH_CHECK(self.defined(), "self is undefined", OPS_ERROR(ErrCode::NOT_SUPPORT));
    
    if (dst.numel() == 0) {
        dst.resize_as_(self);
    }
    TORCH_CHECK(self.sizes() == dst.sizes(),
        "_copy_from_and_resize now only support copy with same size, or dst.numel() == 0!",
        OPS_ERROR(ErrCode::NOT_SUPPORT));
    TORCH_CHECK(self.is_cpu() && dst.device().is_privateuseone(),
        "_copy_from_and_resize now only support copy from cpu tensor to npu tensor, but got src tensor device is ",
        self.device(), " and dst device is ", dst.device(), OPS_ERROR(ErrCode::NOT_SUPPORT));
    dst.copy_(self);
    return dst;
}

}
}
