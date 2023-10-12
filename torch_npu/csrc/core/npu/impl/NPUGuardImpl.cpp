#include <thread>

#include <torch/csrc/jit/serialization/pickler.h>
#include "torch_npu/csrc/core/npu/impl/NPUGuardImpl.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/NPUSerialization.h"

namespace c10_npu {

namespace impl {

constexpr c10::DeviceType NPUGuardImpl::static_type;
thread_local std::once_flag set_device_flag;

void setDeviceOnce() {
  c10_npu::NpuSysCtrl::GetInstance().BackwardsInit();
}

void NPUGuardImpl::setDevice(c10::Device d) const {
  TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1);
  std::call_once(set_device_flag, setDeviceOnce);
}

C10_REGISTER_GUARD_IMPL(PrivateUse1, NPUGuardImpl);

#define REGISTER_PRIVATEUSE1_BACKEND(name)                                                      \
  int rename_privateuse1_backend() {                                                            \
    c10::register_privateuse1_backend(#name);                                                   \
    c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &torch_npu::make_npu_storage_impl); \
    torch::jit::TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1, &torch_npu::npu_info_serialization, &torch_npu::npu_info_deserialization); \
    return 0;                                                                                   \
  }                                                                                             \
  static const int _temp_##name = rename_privateuse1_backend();

REGISTER_PRIVATEUSE1_BACKEND(npu)

} // namespace impl

} // namespace c10_npu
