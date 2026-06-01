#pragma once

#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace c10_npu {
namespace detail {

bool isExternalStream(const NPUStream& stream);

bool isExternalStream(aclrtStream stream, c10::DeviceIndex device_index = -1);

bool isCurrentStreamExternal(c10::DeviceIndex device_index = -1);

void checkNotExternalStream(const NPUStream& stream, const char* api_name);

void checkCurrentStreamNotExternal(c10::DeviceIndex device_index, const char* api_name);

} // namespace detail
} // namespace c10_npu
