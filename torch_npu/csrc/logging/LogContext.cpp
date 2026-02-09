#include "torch_npu/csrc/logging/LogContext.h"
#include <cstdlib>
#include <sstream>

namespace npu_logging {

LogContext &LogContext::GetInstance()
{
    static LogContext instance;
    instance.parseFilterFromEnv();
    return instance;
}

void LogContext::parseFilterFromEnv()
{
    const char* env = std::getenv("TORCH_NPU_LOGS_FILTER");
    if (!env) {
        return;
    }

    is_filter_set_ = true;

    std::string s(env);
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        item = item.substr(item.find_first_not_of(" \t"),
            item.find_last_not_of(" \t") - item.find_first_not_of(" \t") + 1);
        if (item.empty()) continue;
        if (item[0] == '+') {
            has_whitelist_ = true;
            whitelist_.insert(item.substr(1));
        } else if (item[0] == '-') {
            blacklist_.insert(item.substr(1));
        }
    }
}

bool LogContext::shouldLog(const std::string& log_content) const
{
    if (!is_filter_set_) {
        return true;
    }

    for (const auto& keyword : blacklist_) {
        if (log_content.find(keyword) != std::string::npos) {
            return false;
        }
    }

    if (!has_whitelist_) return true;

    for (const auto& keyword : whitelist_) {
        if (log_content.find(keyword) != std::string::npos) {
            return true;
        }
    }

    return false;
}

// Locked from the Outside
void LogContext::GetQNameAndLevelByName(const std::string& name, std::string& qname, LoggingLevel& level)
{
    std::string nameKey = name;
    level = allLevel_;
    qname = "";
    do {
        auto iterLevel = qnameLevels_.find(nameKey);
        if (iterLevel != qnameLevels_.end()) {
            level = static_cast<LoggingLevel>(iterLevel->second);
            qname = iterLevel->first;
            break;
        }
        auto pos = nameKey.rfind('.');
        if (pos == std::string::npos) {
            break;
        }
        nameKey = nameKey.substr(0, pos);
    } while (true);
}

void LogContext::setLogs(const std::unordered_map<std::string, int>& qnameLevels)
{
    std::lock_guard<std::mutex> lock(mutex_);
    qnameLevels_ = qnameLevels;
    auto iter = qnameLevels_.find("torch");
    if (iter != qnameLevels_.end()) {
        allLevel_ = static_cast<LoggingLevel>(iter->second);
    }
    // Global or static logger variables are initialized prior to the invocation of set_logs,
    // the logging levels associated with these loggers should be updated to reflect the new settings.
    for (auto iter = loggers_.begin(); iter != loggers_.end(); iter++) {
        LoggingLevel level = allLevel_;
        std::string qname = iter->second->getQName();
        if (qname.empty()) {
            GetQNameAndLevelByName(iter->first, qname, level);
            iter->second->setQName(qname);
        }
        auto iterLevel = qnameLevels_.find(qname);
        if (iterLevel != qnameLevels_.end()) {
            level = static_cast<LoggingLevel>(iterLevel->second);
        }
        iter->second->setAllowLevel(level);
    }
}

std::shared_ptr<Logger> LogContext::getLogger(const std::string& name) noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = loggers_.find(name);
    if (iter != loggers_.end()) {
        return iter->second;
    }
    std::string qname;
    LoggingLevel level = allLevel_;
    GetQNameAndLevelByName(name, qname, level);
    std::shared_ptr<Logger> logger = std::make_shared<Logger>(name);
    logger->setAllowLevel(level);
    logger->setQName(qname);
    loggers_[name] = logger;
    return logger;
}

}
