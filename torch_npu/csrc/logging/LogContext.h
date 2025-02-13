#ifndef LOGCONTEXT_H
#define LOGCONTEXT_H

#include <mutex>
#include <memory>
#include <list>
#include <unordered_map>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/logging/Logger.h"

namespace npu_logging {
class LogContext {
public:
    LogContext() = default;

    ~LogContext() = default;

    std::shared_ptr<Logger> getLogger(const std::string& name = "");
    static LogContext& GetInstance();
    void setLogs(const std::unordered_map<std::string, int>& aliasLevels);

private:
    void GetAliasAndLevelByName(const std::string& name, std::string& alias, LoggingLevel& level);

    std::mutex mutex_;
    std::unordered_map<std::string, int> aliasLevels_;
    LoggingLevel allLevel_ = LoggingLevel::WARNING;
    std::unordered_map<std::string, std::shared_ptr<Logger>> loggers_;
};

C10_NPU_API inline LogContext& logging()
{
    return LogContext::GetInstance();
}

}

#endif
