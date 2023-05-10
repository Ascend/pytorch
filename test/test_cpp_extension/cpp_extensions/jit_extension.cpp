// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
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

#include <torch/extension.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>

#include "doubler.h"

using namespace at;

Tensor exp_add(Tensor x, Tensor y);

Tensor tanh_add(Tensor x, Tensor y) {
  return x.tanh() + y.tanh();
}

Tensor npu_add(const Tensor& self_, const Tensor& other_) {
  return at_npu::native::NPUNativeFunctions::add(self_, other_, 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tanh_add", &tanh_add, "tanh(x) + tanh(y)");
  m.def("exp_add", &exp_add, "exp(x) + exp(y)");
  m.def("npu_add", &npu_add, "x + y");
  py::class_<Doubler>(m, "Doubler")
  .def(py::init<int, int>())
  .def("forward", &Doubler::forward)
  .def("get", &Doubler::get);
}
