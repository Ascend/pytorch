#include "torch_npu/csrc/logging/LogContext.h"

namespace npu_logging {

LogContext &LogContext::GetInstance()
{
    static LogContext instance;
    return instance;
}

// Locked from the Outside
void LogContext::GetAliasAndLevelByName(const std::string& name, std::string& alias, LoggingLevel& level)
{
    std::string nameKey = name;
    level = allLevel_;
    alias = "";
    do {
        auto iterLevel = aliasLevels_.find(nameKey);
        if (iterLevel != aliasLevels_.end()) {
            level = static_cast<LoggingLevel>(iterLevel->second);
            alias = iterLevel->first;
            break;
        }
        auto pos = nameKey.rfind('.');
        if (pos == std::string::npos) {
            break;
        }
        nameKey = nameKey.substr(0, pos);
    } while (true);
}

void LogContext::setLogs(const std::unordered_map<std::string, int>& aliasLevels)
{
    std::lock_guard<std::mutex> lock(mutex_);
    aliasLevels_ = aliasLevels;
    auto iter = aliasLevels_.find("torch");
    if (iter != aliasLevels_.end()) {
        allLevel_ = static_cast<LoggingLevel>(iter->second);
    }
    for (auto iter = loggers_.begin(); iter != loggers_.end(); iter++) {
        LoggingLevel level = allLevel_;
        std::string alias = iter->second->getModuleAlias();
        if (alias.empty()) {
            GetAliasAndLevelByName(iter->first, alias, level);
            iter->second->setModuleAlias(alias);
        }
        auto iterLevel = aliasLevels_.find(alias);
        if (iterLevel != aliasLevels_.end()) {
            level = static_cast<LoggingLevel>(iterLevel->second);
        }
        iter->second->setAllowLevel(level);
    }
}

std::shared_ptr<Logger> LogContext::getLogger(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = loggers_.find(name);
    if (iter != loggers_.end()) {
        return iter->second;
    }
    std::string alias;
    LoggingLevel level = allLevel_;
    GetAliasAndLevelByName(name, alias, level);
    std::shared_ptr<Logger> logger = std::make_shared<Logger>(name);
    logger->setAllowLevel(level);
    logger->setModuleAlias(alias);
    loggers_[name] = logger;
    return logger;
}

}
