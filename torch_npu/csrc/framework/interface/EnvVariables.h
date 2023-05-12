#ifndef __PLUGIN_NATIVE_NPU_INTERFACE_ENVVARIABLES__
#define __PLUGIN_NATIVE_NPU_INTERFACE_ENVVARIABLES__

namespace at_npu {
namespace native {
namespace env {

/**
  check if the autotuen is enabled, return true or false.
  */
bool AutoTuneEnabled();
bool CheckBmmV2Enable();
bool CheckJitDisable();
bool CheckProfilingEnable();
bool CheckMmBmmNDDisable();
bool CheckForbidInternalFormat();

} // namespace env
} // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_INTERFACE_ENVVARIABLES__
