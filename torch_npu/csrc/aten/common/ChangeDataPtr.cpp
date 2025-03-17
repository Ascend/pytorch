#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace native {

int64_t NPUNativeFunctions::npu_change_data_ptr(const at::Tensor& dst, const at::Tensor& src, int64_t index)
{
    TORCH_CHECK(
        index >= 0,
        "Expect offset(index) equal or greater than zero, got: ", index, PTA_ERROR(ErrCode::VALUE));

    const auto& src_scalar_type = src.scalar_type();
    const auto& dst_scalar_type = dst.scalar_type();

    TORCH_CHECK(
        src_scalar_type == dst_scalar_type,
        "Expect src and dst tensors having the same dtype, got: ",
        "src with dtype ", src_scalar_type,
        ", dst with dtype ", dst_scalar_type, PTA_ERROR(ErrCode::TYPE));
    TORCH_CHECK(
        (src_scalar_type == at::ScalarType::Half) ||
        (src_scalar_type == at::ScalarType::Float) ||
        (src_scalar_type == at::ScalarType::BFloat16),
        "Only supports src and dst tensors with dtype float32, float16 or bfloat16, got: ", src_scalar_type,
        PTA_ERROR(ErrCode::TYPE));

    auto dst_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_.storage_sizes_;
    auto src_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_.storage_sizes_;
    int64_t dst_storage_size = c10::multiply_integers(dst_sizes);
    int64_t src_storage_size = c10::multiply_integers(src_sizes);

    TORCH_CHECK(
        index + dst_storage_size * dst.element_size() <=
        src_storage_size * src.element_size(),
        "Offsets overflow, got: ",
        "offset(index) ", index,
        ", dst storage size ", dst_storage_size,
        ", src storage size ", src_storage_size, PTA_ERROR(ErrCode::PARAM));

    at::DataPtr aim_data_ptr;
    if (src_scalar_type == at::ScalarType::Float) {
        float* data_ptr = static_cast<float*>(src.storage().data_ptr().get()) + index;
        aim_data_ptr = at::DataPtr(data_ptr, dst.storage().device());
    } else if (src_scalar_type == at::ScalarType::BFloat16) {
        at::BFloat16* data_ptr = static_cast<at::BFloat16*>(src.storage().data_ptr().get()) + index;
        aim_data_ptr = at::DataPtr(data_ptr, dst.storage().device());
    } else {
        at::Half* data_ptr = static_cast<at::Half*>(src.storage().data_ptr().get()) + index;
        aim_data_ptr = at::DataPtr(data_ptr, dst.storage().device());
    }
    dst.storage().set_data_ptr(std::move(aim_data_ptr));

    return 0;
}

} // namespace native
} // namespace at_npu
