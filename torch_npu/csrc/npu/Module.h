// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#ifndef THNP_NPU_MODULE_INC
#define THNP_NPU_MODULE_INC

void THNPModule_setDevice(int idx);
void RegisterNPUDeviceProperties(PyObject *module);
void BindGetDeviceProperties(PyObject *module);
PyObject *THNPModule_getDevice_wrap(PyObject *self);
PyObject *THNPModule_setDevice_wrap(PyObject *self, PyObject *arg);
PyObject *THNPModule_getDeviceName_wrap(PyObject *self, PyObject *arg);
PyObject *THNPModule_getDriverVersion(PyObject *self);
PyObject *THNPModule_isDriverSufficient(PyObject *self);
PyObject *THNPModule_getCurrentBlasHandle_wrap(PyObject *self);
PyObject *THNPModule_npu_datadump_enable(PyObject *self, PyObject *args);
PyObject *THNPModule_npu_datadump_disable(PyObject *self, PyObject *noargs);

#define CHANGE_UNIT_SIZE 1024.0
#endif
