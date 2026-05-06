#pragma once
#include "torch_npu/csrc/logging/LogContext.h"

inline std::shared_ptr<npu_logging::Logger>& GetRecoveryLogger()
{
    static std::shared_ptr<npu_logging::Logger> loggerRecovery = npu_logging::logging().getLogger("torch_npu.recovery");
    return loggerRecovery;
}

#define TORCH_NPU_RECOVERY_LOGI(format, ...)                          \
    do {                                                              \
        TORCH_NPU_LOGI(GetRecoveryLogger(), format, ##__VA_ARGS__);   \
        ASCEND_LOGI(format, ##__VA_ARGS__);                           \
    } while (0);
