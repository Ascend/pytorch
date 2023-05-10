// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <c10/core/TensorOptions.h>
#include <torch_npu/csrc/utils/Utils.h>
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

namespace torch_npu {
namespace utils {

void npu_lazy_init();

void npu_set_run_yet_variable_to_false();

static void maybe_initialize_npu(const at::TensorOptions& options) {
  if (torch_npu::utils::is_npu(options)) {
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
  if (torch_npu::utils::is_npu(device)) {
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