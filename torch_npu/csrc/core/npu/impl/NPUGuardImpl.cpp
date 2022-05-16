#include "torch_npu/csrc/core/npu/impl/NPUGuardImpl.h"

namespace c10_npu {

namespace impl {

constexpr c10::DeviceType NPUGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(XLA, NPUGuardImpl);

} // namespace impl

} // namespace c10_npu
