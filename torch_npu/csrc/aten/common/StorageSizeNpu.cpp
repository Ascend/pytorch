#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"

namespace at_npu {
namespace native {

int64_t NPUNativeFunctions::get_storage_size(const at::Tensor& self)
{
    torch_npu::utils::torch_check_npu(self);
    auto sizes = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.storage_sizes_;
    int64_t n = 1;
    for (auto s : sizes) {
        n *= s;
    }
    return n;
}

int64_t NPUNativeFunctions::get_storage_base_nbytes(const at::Tensor& self)
{
    torch_npu::utils::torch_check_npu(self);
    auto base_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.base_sizes_;
    int64_t n = 1;
    for (auto s : base_sizes) {
        n *= s;
    }
    return n * self.element_size();
}

} // namespace native
} // namespace at_npu
