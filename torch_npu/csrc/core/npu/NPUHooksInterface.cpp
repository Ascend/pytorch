#include "torch_npu/csrc/core/npu/NPUHooksInterface.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/aten/common/ResizeNpu.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/utils/LazyInit.h"
#endif

namespace c10_npu {

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, NPUHooksInterface, NPUHooksArgs);
#define REGISTER_PRIVATEUSE1_HOOKS(clsname) \
  C10_REGISTER_CLASS(PrivateUse1HooksRegistry, clsname, clsname)

C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, NPUHooksInterface, NPUHooksArgs)

void NPUHooksInterface::initPrivateUse1() const
{
#ifndef BUILD_LIBTORCH
    torch_npu::utils::npu_lazy_init();
#endif
}

bool NPUHooksInterface::hasPrimaryContext(c10::DeviceIndex device_index) const
{
    return c10_npu::isDeviceCtxActive(device_index);
}

void NPUHooksInterface::resizePrivateUse1Bytes(const c10::Storage &storage, size_t new_bytes) const
{
    auto storage_impl = static_cast<torch_npu::NPUStorageImpl*>(storage.unsafeGetStorageImpl());
    auto format = storage_impl->npu_desc_.npu_format_;
    TORCH_CHECK(at_npu::native::FormatHelper::IsBaseFormatType(format),
                "Try to resize a storage without base format",
                PTA_ERROR(ErrCode::TYPE));

    auto itemsize = storage_impl->npu_desc_.data_type_.itemsize();
    TORCH_CHECK(itemsize > 0,
                "Try to resize a storage with data_type.itemsize <= 0",
                PTA_ERROR(ErrCode::TYPE));
    std::vector<int64_t> new_size = {static_cast<int64_t>(new_bytes) / (ptrdiff_t)itemsize};
    at_npu::native::storage_resize_npu(*storage_impl, new_bytes, new_size, true);
}

at::PrivateUse1HooksInterface* get_npu_hooks()
{
    static at::PrivateUse1HooksInterface* npu_hooks;
    static c10::once_flag once;
    c10::call_once(once, [] {
        npu_hooks = new NPUHooksInterface();
    });
    return npu_hooks;
}
}
