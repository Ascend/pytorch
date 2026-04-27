#ifndef THNP_NPU_MODULE_INC
#define THNP_NPU_MODULE_INC
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/logging/LogContext.h"

inline std::shared_ptr<npu_logging::Logger>& GetRecoveryLogger()
{
    static std::shared_ptr<npu_logging::Logger> loggerRecovery = npu_logging::logging().getLogger("torch_npu.recovery");
    return loggerRecovery;
}

#define TORCH_NPU_RECOVERY_LOGI(format, ...)                                     \
    do {                                                                         \
        TORCH_NPU_LOGI(GetRecoveryLogger(), format, ##__VA_ARGS__);            \
        ASCEND_LOGI(format, ##__VA_ARGS__);                                      \
    } while (0);

void THNPModule_setDevice(int idx);
TORCH_NPU_API void RegisterNPUDeviceProperties(PyObject *module);
TORCH_NPU_API void BindGetDeviceProperties(PyObject *module);
TORCH_NPU_API void RegisterNPUDeviceMemories(PyObject *module);
TORCH_NPU_API void BindGetDeviceMemories(PyObject *module);
TORCH_NPU_API void RegisterNpuPluggableAllocator(PyObject *module);
TORCH_NPU_API void initCommMethods();
PyObject *THNPModule_getDevice_wrap(PyObject *self);
PyObject *THNPModule_setDevice_wrap(PyObject *self, PyObject *arg);
PyObject *THNPModule_getDeviceName_wrap(PyObject *self, PyObject *arg);
PyObject *THNPModule_getDriverVersion(PyObject *self);
PyObject *THNPModule_isDriverSufficient(PyObject *self);
PyObject *THNPModule_getCurrentBlasHandle_wrap(PyObject *self);

#define CHANGE_UNIT_SIZE 1024.0
#endif
