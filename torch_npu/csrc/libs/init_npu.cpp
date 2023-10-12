#include "torch_npu/csrc/libs/init_npu.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"


namespace torch_npu {

bool is_npu_device(const at::Device& device) {
  return device.type() == c10::DeviceType::PrivateUse1;
}


void init_npu(const c10::DeviceIndex device_index) {
  c10_npu::NpuSysCtrl::SysStatus status =
      c10_npu::NpuSysCtrl::GetInstance().Initialize((int)device_index);
  if (status != c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
    C10_NPU_SHOW_ERR_MSG();
    return;
  }
}


void init_npu(const std::string& device_str) {
  auto device = at::Device(device_str);
  TORCH_CHECK(is_npu_device(device), "NPU device init fail, except got NPU device, but got ", device_str);
  init_npu(device.index());
}


void init_npu(const at::Device& device) {
  TORCH_CHECK(is_npu_device(device), "NPU device init fail, except got NPU device, but got ", str(device));
  init_npu(device.index());
}

} // namespace torch_npu


namespace torch {
namespace npu {

void synchronize(int64_t device_index) {
  c10_npu::NPUGuard device_guard(at::Device(at::DeviceType::PrivateUse1, device_index));
  c10_npu::npuSynchronizeDevice();
}

} // namespace npu
} // namespace torch


namespace c10 {
namespace npu {

DeviceIndex current_device() {
  if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
    int device;
    aclrtGetDevice(&device);
    return (c10::DeviceIndex)device;
  } else {
    TORCH_NPU_WARN("Please init npu device first!");
    return (c10::DeviceIndex)-1;
  }
}

} // namespace npu
} // namespace c10
