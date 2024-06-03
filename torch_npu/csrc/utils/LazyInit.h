#pragma once
#include <pybind11/pybind11.h>
#include <c10/core/TensorOptions.h>
#include <torch/csrc/Exceptions.h>
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

namespace torch_npu {
namespace utils {

void npu_lazy_init();

void npu_set_run_yet_variable_to_false();

static bool isNPUDevice(const at::TensorOptions& options) {
    return options.device().type() == at_npu::key::NativeDeviceType;
}

static bool isNPUDevice(const at::Device& device) {
  return device.type() == at_npu::key::NativeDeviceType;
}

static void maybe_initialize_npu(const at::TensorOptions& options) {
  if (isNPUDevice(options)) {
    {
      pybind11::gil_scoped_release no_gil;
      c10_npu::NpuSysCtrl::SysStatus status =
          c10_npu::NpuSysCtrl::GetInstance().Initialize(options.device().index());
      if (status !=
          c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
        throw python_error();
      }
    }
    torch_npu::utils::npu_lazy_init();
  }
}

static void maybe_initialize_npu(const at::Device& device) {
  if (isNPUDevice(device)) {
    {
      pybind11::gil_scoped_release no_gil;
      c10_npu::NpuSysCtrl::SysStatus status =
          c10_npu::NpuSysCtrl::GetInstance().Initialize(device.index());
      if (status !=
          c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
        throw python_error();
      }
    }
    torch_npu::utils::npu_lazy_init();
  }
}

static void maybe_initialize_npu(const c10::optional<at::Device>& device) {
  if (device) {
    maybe_initialize_npu(*device);
  }
}

}
}