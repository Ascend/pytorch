#pragma once
#include <c10/core/TensorOptions.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#ifndef BUILD_LIBTORCH
#include "torch_npu/csrc/utils/LazyInit.h"
#endif


namespace torch_npu {
namespace utils {

inline bool is_npu(const at::Tensor& tensor) {
    return tensor.is_privateuseone();
}

inline bool is_npu(const at::TensorOptions& options) {
    return options.device().is_privateuseone();
}

inline bool is_npu(const at::Device& device) {
    return device.is_privateuseone();
}

inline void torch_check_npu(const at::Tensor& tensor) {
  TORCH_CHECK(is_npu(tensor),
              "Expected NPU tensor, please check whether the input tensor device is correct.",
              PTA_ERROR(ErrCode::PARAM));
}

inline void torch_check_npu(const at::TensorOptions& options) {
  TORCH_CHECK(is_npu(options),
              "Expected NPU tensor, please check whether the input tensor device is correct.",
              PTA_ERROR(ErrCode::PARAM));
}

inline void torch_check_npu(const at::Device& device) {
  TORCH_CHECK(is_npu(device),
              "Expected NPU tensor, please check whether the input tensor device is correct.",
              PTA_ERROR(ErrCode::PARAM));
}

inline c10::DeviceType get_npu_device_type() {
  return c10::DeviceType::PrivateUse1;
}

inline void maybe_initialize_npu(const at::TensorOptions& options) {
  if (torch_npu::utils::is_npu(options)) {
    c10_npu::NpuSysCtrl::SysStatus status =
        c10_npu::NpuSysCtrl::GetInstance().Initialize(options.device().index());
    if (status != c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
      TORCH_CHECK(false, "npu device ", options.device().index(), " init failed.",
                  PTA_ERROR(ErrCode::INTERNAL));
    }
#ifndef BUILD_LIBTORCH
    torch_npu::utils::npu_lazy_init();
#endif
  }
}

inline void maybe_initialize_npu(const at::Device& device) {
  if (torch_npu::utils::is_npu(device)) {
    c10_npu::NpuSysCtrl::SysStatus status =
        c10_npu::NpuSysCtrl::GetInstance().Initialize(device.index());
    if (status != c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
      TORCH_CHECK(false, "npu device ", device.index(), " init failed.",
                  PTA_ERROR(ErrCode::INTERNAL));
    }
#ifndef BUILD_LIBTORCH
    torch_npu::utils::npu_lazy_init();
#endif
  }
}

inline void maybe_initialize_npu(const c10::optional<at::Device>& device) {
  if (device) {
    maybe_initialize_npu(*device);
  }
}

}
}
