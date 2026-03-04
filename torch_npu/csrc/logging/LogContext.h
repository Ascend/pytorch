#ifndef LOGCONTEXT_H
#define LOGCONTEXT_H

#include <mutex>
#include <memory>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <string>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/logging/Logger.h"
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace npu_logging {
class LogContext {
public:
    LogContext() = default;

    ~LogContext() = default;

    std::shared_ptr<Logger> getLogger(const std::string& name = "") noexcept;
    static LogContext& GetInstance();
    void setLogs(const std::unordered_map<std::string, int>& qnameLevels);
    bool shouldLog(const std::string& log_content) const;
    void parseFilterFromEnv();

private:
    void GetQNameAndLevelByName(const std::string& name, std::string& qname, LoggingLevel& level);

    std::mutex mutex_;
    std::unordered_map<std::string, int> qnameLevels_;
    LoggingLevel allLevel_ = LoggingLevel::WARNING;
    std::unordered_map<std::string, std::shared_ptr<Logger>> loggers_;

    std::unordered_set<std::string> whitelist_;
    std::unordered_set<std::string> blacklist_;
    bool has_whitelist_ = false;
    bool is_filter_set_ = false;
};

C10_NPU_API inline LogContext& logging()
{
    return LogContext::GetInstance();
}

C10_NPU_API inline bool should_log(const std::string& log_content) noexcept
{
    return LogContext::GetInstance().shouldLog(log_content);
}

static std::shared_ptr<Logger> loggerMem = logging().getLogger("torch_npu.memory");
static std::shared_ptr<Logger> loggerHccl = logging().getLogger("torch.distributed");

} // namespace npu_logging

// macros for log memory
#define TORCH_NPU_MEMORY_LOGD(format, ...)                                     \
    do {                                                                       \
        TORCH_NPU_LOGD(npu_logging::loggerMem, format, ##__VA_ARGS__);         \
        ASCEND_LOGD(format, ##__VA_ARGS__);                                    \
    } while (0);

#define TORCH_NPU_MEMORY_LOGI(format, ...)                                     \
    do {                                                                       \
        TORCH_NPU_LOGI(npu_logging::loggerMem, format, ##__VA_ARGS__);         \
        ASCEND_LOGI(format, ##__VA_ARGS__);                                    \
    } while (0);

#define TORCH_NPU_MEMORY_LOGW(format, ...)                                     \
    do {                                                                       \
        TORCH_NPU_LOGW(npu_logging::loggerMem, format, ##__VA_ARGS__);         \
        ASCEND_LOGW(format, ##__VA_ARGS__);                                    \
    } while (0);

#define TORCH_NPU_MEMORY_LOGE(format, ...)                                     \
    do {                                                                       \
        TORCH_NPU_LOGE(npu_logging::loggerMem, format, ##__VA_ARGS__);         \
        ASCEND_LOGE(format, ##__VA_ARGS__);                                    \
    } while (0);
        
#define TORCH_NPU_MEMORY_LOGC(format, ...)                                     \
    do {                                                                       \
        TORCH_NPU_LOGC(npu_logging::loggerMem, format, ##__VA_ARGS__);         \
        ASCEND_LOGE(format, ##__VA_ARGS__);                                    \
    } while (0);

// macros for log hccl
#define TORCH_NPU_HCCL_LOGD(format, ...)                                       \
    do {                                                                       \
        TORCH_NPU_LOGD(npu_logging::loggerHccl, format, ##__VA_ARGS__);        \
        ASCEND_LOGD(format, ##__VA_ARGS__);                                    \
    } while (0);

#define TORCH_NPU_HCCL_LOGI(format, ...)                                       \
    do {                                                                       \
        TORCH_NPU_LOGI(npu_logging::loggerHccl, format, ##__VA_ARGS__);        \
        ASCEND_LOGI(format, ##__VA_ARGS__);                                    \
    } while (0);

#define TORCH_NPU_HCCL_LOGW(format, ...)                                       \
    do {                                                                       \
        TORCH_NPU_LOGW(npu_logging::loggerHccl, format, ##__VA_ARGS__);        \
        ASCEND_LOGW(format, ##__VA_ARGS__);                                    \
    } while (0);

#define TORCH_NPU_HCCL_LOGE(format, ...)                                       \
    do {                                                                       \
        TORCH_NPU_LOGE(npu_logging::loggerHccl, format, ##__VA_ARGS__);        \
        ASCEND_LOGE(format, ##__VA_ARGS__);                                    \
    } while (0);

#define TORCH_NPU_HCCL_LOGC(format, ...)                                       \
    do {                                                                       \
        TORCH_NPU_LOGC(npu_logging::loggerHccl, format, ##__VA_ARGS__);        \
        ASCEND_LOGE(format, ##__VA_ARGS__);                                    \
    } while (0);

#endif
