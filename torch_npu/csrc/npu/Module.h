#ifndef THNP_NPU_MODULE_INC
#define THNP_NPU_MODULE_INC
#include "torch_npu/csrc/core/npu/NPUMacros.h"

void THNPModule_setDevice(int idx);
TORCH_NPU_API void RegisterNPUDeviceProperties(PyObject *module);
TORCH_NPU_API void BindGetDeviceProperties(PyObject *module);
TORCH_NPU_API void RegisterNPUDeviceMemories(PyObject *module);
TORCH_NPU_API void BindGetDeviceMemories(PyObject *module);
PyObject *THNPModule_getDevice_wrap(PyObject *self);
PyObject *THNPModule_setDevice_wrap(PyObject *self, PyObject *arg);
PyObject *THNPModule_getDeviceName_wrap(PyObject *self, PyObject *arg);
PyObject *THNPModule_getDriverVersion(PyObject *self);
PyObject *THNPModule_isDriverSufficient(PyObject *self);
PyObject *THNPModule_getCurrentBlasHandle_wrap(PyObject *self);

#define CHANGE_UNIT_SIZE 1024.0
#endif
