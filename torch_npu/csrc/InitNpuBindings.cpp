// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include <Python.h>
#include <ATen/Parallel.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>

#include "torch_npu/csrc/npu/Event.h"
#include "torch_npu/csrc/npu/ReplayFunctions.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/graph/execute/GraphExecutor.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"

#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"
#include "torch_npu/csrc/distributed/Init.h"
#include "torch_npu/csrc/profiler/init.h"
#include "torch_npu/csrc/npu/Generator.h"
#include "torch_npu/csrc/npu/Module.h"
#include "torch_npu/csrc/utils/TensorMethods.h"

PyObject* module;


void AddPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods)
{
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

PyObject * THPModule_npu_shutdown(PyObject * /* unused */)
{
  // cudaFree is blocking and will synchronize across all kernels executing
  // on the current device, while aclrtFree Free device memory immediately.
  // aclrtSynchronizeDevice should be called before aclrtFree to ensure that
  // all of op tasks completed before device memory free.
  if (c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
    c10_npu::npuSynchronizeDevice();
    at_npu::native::GraphExecutor::GetInstance().Finalize();
    THNPUCachingHostAllocator_emptyCache();
    c10_npu::NPUCachingAllocator::emptyCache();
    c10_npu::NpuSysCtrl::SysStatus status = c10_npu::NpuSysCtrl::GetInstance().Finalize();
    if (status != c10_npu::NpuSysCtrl::SysStatus::FINALIZE_SUCC) {
      fprintf(stdout, "THPModule_npu_shutdown failed.\n");
    } else {
      fprintf(stdout, "THPModule_npu_shutdown success.\n");
    }
  }
  Py_RETURN_NONE;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
static PyMethodDef TorchNpuMethods[] = {
  {"_npu_shutdown", (PyCFunction)THPModule_npu_shutdown, METH_NOARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

void THNPStream_init(PyObject *module);
void THNPEvent_init(PyObject *module);
void THNPReplayGraph_init(PyObject *module);
bool THPGenerator_init(PyObject *module);
PyMethodDef* THNPModule_get_methods();

namespace torch_npu { namespace autograd {
  void initTorchFunctions(PyObject *module);
}
}

static std::vector<PyMethodDef> methods;

extern "C"

PyObject* initModule(){
  at::internal::lazy_init_num_threads();

  AddPyMethodDefs(methods, TorchNpuMethods);
  AddPyMethodDefs(methods, THNPModule_get_methods());
  AddPyMethodDefs(methods, torch_npu::profiler::profiler_functions());
  AddPyMethodDefs(methods, torch_npu::distributed::python_functions());
  AddPyMethodDefs(methods, torch_npu::utils::tensor_functions());
  static struct PyModuleDef torchnpu_module = {
     PyModuleDef_HEAD_INIT,
     "torch_npu._C",
     nullptr,
     -1,
     methods.data()
  };
  module = PyModule_Create(&torchnpu_module);

  // This will only initialize base classes and attach them to library namespace
  // They won't be ready for real usage until importing npu module, that will
  // complete the process (but it defines Python classes before calling back into
  // C, so these lines have to execute first)..
  THNPStream_init(module);
  THNPEvent_init(module);
  THNPReplayGraph_init(module);
  THPGenerator_init(module);

  torch_npu::autograd::initTorchFunctions(module);

  RegisterNPUDeviceProperties(module);
  BindGetDeviceProperties(module);
  return module;
}

PyMODINIT_FUNC PyInit__C(void){
  return initModule();
}