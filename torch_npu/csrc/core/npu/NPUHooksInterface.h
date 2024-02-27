#pragma once
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"

namespace c10_npu {

struct TORCH_API NPUHooksInterface : public at::PrivateUse1HooksInterface {
    virtual ~NPUHooksInterface() = default;
    const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
      static auto device_gen = at_npu::detail::getDefaultNPUGenerator(device_index);
      return device_gen;
    }
    void initPrivateUse1() const override;
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override;
};

struct TORCH_API NPUHooksArgs : public at::PrivateUse1HooksArgs {};

// register to PrivateUse1HooksInterface
at::PrivateUse1HooksInterface* get_npu_hooks();
}
