// Copyright (c) 2022 Huawei Technologies Co., Ltd
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


#include "torch_npu/csrc/utils/TensorMethods.h"

namespace torch_npu {
namespace utils {

const char* _backend_to_string_npu(const at::Backend& backend) {
  switch (backend) {
    case at::Backend::CPU: return "torch";
    case at_npu::key::NativeBackend: return "torch.npu";
    default: AT_ERROR("Unimplemented backend ", backend);
  }
}

static PyObject * THPVariable_npu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "npu(Tensor temp, Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "npu(Tensor temp, Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  auto device = r.isNone(1) ? c10::Device(at_npu::key::NativeDeviceType) : r.device(1);
  auto opt_memory_format = r.memoryformatOptional(3);
  TORCH_CHECK((device.type() == at_npu::key::NativeDeviceType), "Invalid device, must be npu device");
  maybe_initialize_npu(device);
  pybind11::gil_scoped_release no_gil;
  return THPVariable_Wrap(self_.to(device, r.toBool(2), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_is_npu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "type(Tensor temp)"
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  return torch::autograd::utils::wrap(at_npu::key::isDeviceTensor(self_));
  END_HANDLE_TH_ERRORS
}

// autograd methods on torch._C
static PyMethodDef TorchTensorMethods[] = { // NOLINT
  {"npu", castPyCFunctionWithKeywords(THPVariable_npu), METH_VARARGS | METH_KEYWORDS, NULL},
  {"is_npu", castPyCFunctionWithKeywords(THPVariable_is_npu), METH_VARARGS | METH_KEYWORDS, NULL},
  {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* tensor_functions() {
  return TorchTensorMethods;
}

}
}