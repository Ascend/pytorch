#include "hardware/cpu/cpu_device_context.h"
#include <map>
#include <string>
#include <unordered_set>
#include <utility>

#include "hardware/hardware_abstract/device_context_manager.h"

namespace fxrt {
namespace device {
namespace cpu {
namespace {
const char kCPUDevice[] = "CPU";

} // namespace

void CPUDeviceContext::Initialize() {
  if (initialized_) {
    return;
  }
  deviceResManager_->Initialize();
  initialized_ = true;
}

void CPUDeviceContext::Destroy() {
  deviceResManager_->Destroy();
  initialized_ = false;
}

// Register functions to _c_expression so python hal module could call CPU device interfaces.
FXRT_REGISTER_DEVICE(kCPUDevice, CPUDeviceContext);
} // namespace cpu
} // namespace device
} // namespace fxrt
