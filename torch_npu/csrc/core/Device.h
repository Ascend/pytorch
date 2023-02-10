// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context

#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <ATen/Device.h>


// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct TORCH_API TNPDevice {
  PyObject_HEAD
  at::Device device;
};

TORCH_API extern PyTypeObject TNPDeviceType;

inline bool TNPDevice_Check(PyObject *obj) {
  return Py_TYPE(obj) == &TNPDeviceType;
}

PyObject * TNPDevice_New(const at::Device& device);

void TNPDevice_init(PyObject *module);
