#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"

namespace fxrt {
namespace device {
DeviceResManager::DeviceResManager() {
  deviceContext_ = nullptr;
}

bool DeviceContext::initialized() const {
  return initialized_;
}

DeviceContextKey DeviceToDeviceContextKey(hardware::Device device) {
  uint32_t deviceId = static_cast<uint32_t>(std::max(static_cast<int32_t>(0), static_cast<int32_t>(device.index)));
  if (deviceId != collective::CollectiveManager::Instance().local_rank_id() &&
      device.type != fxrt::hardware::DeviceType::CPU) {
    RT_GLOG(EXCEPTION) << "Device id: " << deviceId << " is not equal to CollectiveManager local_rank_id: "
                       << collective::CollectiveManager::Instance().local_rank_id();
  }
  return {hardware::GetDeviceNameByType(device.type), deviceId};
}
} // namespace device
} // namespace fxrt
