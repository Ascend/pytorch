// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BUILD_LIBTORCH
#include <torch/csrc/utils/pybind.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "op_plugin/ops/dvm/lazy_fusion_control.h"

void TORCH_NPU_API THNPLazyFusion_init(PyObject* module) {
  auto torch_C_m = pybind11::handle(module).cast<pybind11::module>();
  auto fusion_m = torch_C_m.def_submodule("_lazy_fusion", "Eager lazy fusion controls");
  fusion_m.def(
      "_set_disabled",
      &lazy_fusion::SetLazyFusionDisabled,
      pybind11::arg("disabled"),
      "Set the script-side disable gate and return its previous state.");
  fusion_m.def(
      "_set_dump_enabled",
      &lazy_fusion::SetLazyFusionDumpEnabled,
      pybind11::arg("enabled"),
      "Set the script-side dump gate and return its previous state.");
}
#endif
