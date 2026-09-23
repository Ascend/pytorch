#ifndef __HARDWARE_DEVICE_H__
#define __HARDWARE_DEVICE_H__

#include <cstdint>
#include <vector>

#include "common/common.h"

namespace fxrt {
namespace hardware {
const std::vector<std::string> deviceNames = {"CPU", "Ascend"};

/**
 * @brief Enumeration of supported device types.
 */
enum class DeviceType : int8_t {
  CPU = 0, ///< CPU device
  NPU = 1, ///< NPU device
           // Add other device types here
};

inline const std::string& GetDeviceNameByType(const DeviceType& type) {
  auto deviceType = static_cast<int8_t>(type);
  CHECK_IF_FAIL(deviceType >= 0);
  CHECK_IF_FAIL(static_cast<size_t>(deviceType) < deviceNames.size());
  return deviceNames[deviceType];
}

/**
 * @brief Represents a specific device.
 */
using DeviceIndex = int8_t;

/**
 * @brief Represents a compute device.
 */
struct Device {
  DeviceType type = DeviceType::CPU; ///< The type of the device.
  DeviceIndex index = -1; ///< The index of the device, -1 for any index.

  /**
   * @brief Constructs a Device.
   * @param t The device type.
   * @param i The device index.
   */
  Device(DeviceType t, DeviceIndex i = -1) : type(t), index(i) {}

  /**
   * @brief Equality comparison operator.
   * @param other The other Device to compare with.
   * @return true if the devices are the same, false otherwise.
   */
  bool operator==(const Device& other) const {
    return type == other.type && index == other.index;
  }
  /**
   * @brief Inequality comparison operator.
   * @param other The other Device to compare with.
   * @return true if the devices are different, false otherwise.
   */
  bool operator!=(const Device& other) const {
    return !(*this == other);
  }
};

} // namespace hardware
} // namespace fxrt

#endif // __HARDWARE_DEVICE_H__
