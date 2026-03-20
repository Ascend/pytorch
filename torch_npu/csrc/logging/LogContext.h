#ifndef LOGCONTEXT_H
#define LOGCONTEXT_H

#include <mutex>
#include <memory>
#include <list>
#include <unordered_map>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/logging/Logger.h"
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace npu_logging {
class TORCH_NPU_API LogContext {
public:
    LogContext() = default;

    ~LogContext() = default;

    std::shared_ptr<Logger> getLogger(const std::string& name = "") noexcept;
    static LogContext& GetInstance();
    void setLogs(const std::unordered_map<std::string, int>& qnameLevels);

private:
    void GetQNameAndLevelByName(const std::string& name, std::string& qname, LoggingLevel& level);

    std::mutex mutex_;
    std::unordered_map<std::string, int> qnameLevels_;
    LoggingLevel allLevel_ = LoggingLevel::WARNING;
    std::unordered_map<std::string, std::shared_ptr<Logger>> loggers_;
};

C10_NPU_API inline LogContext& logging()
{
    return LogContext::GetInstance();
}

} // namespace npu_logging

#endif
