#include <iostream>
#include <cstdio>
#include <ctime>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include "torch_npu/csrc/logging/Logger.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace npu_logging {

static std::unordered_map<LoggingLevel, std::string> LoggingLevelNames = {
    {LoggingLevel::DEBUG, "DEBUG"},
    {LoggingLevel::INFO, "INFO"},
    {LoggingLevel::WARNING, "WARNING"},
    {LoggingLevel::ERROR, "ERROR"},
    {LoggingLevel::CRITICAL, "CRITICAL"},
};

void Logger::setAllowLevel(LoggingLevel level)
{
    allow_level_ = level;
}

void Logger::setModuleAlias(const std::string& alias)
{
    alias_ = alias;
}

std::string Logger::getModuleAlias()
{
    return alias_;
}

void Logger::log(LoggingLevel level, const char* format, va_list args)
{
    const int log_buffer_size = 1024;
    char buffer[log_buffer_size] = {0};

    int ret = vsnprintf(buffer, log_buffer_size, format, args);
    if (ret < 0) {
        return;
    }
    struct timespec ts = {0};
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm = {0};
    localtime_r(&ts.tv_sec, &tm);

    char timeBuffer[64] = {0};
    std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", &tm);

    long nowMs = ts.tv_nsec / 1000000;

    auto rank = c10_npu::option::OptionsManager::GetRankId();
    if (rank != -1) {
        std::cerr << "[rank:" << rank << "]:[" << timeBuffer << ":" << std::setfill('0') << std::setw(3) << nowMs << "] " << name_ <<
            ": [" << LoggingLevelNames[level] << "] " << buffer << std::endl;
    } else {
        std::cerr << "[" << timeBuffer << ":" << std::setfill('0') << std::setw(3) << nowMs << "] " << name_ << ": [" <<
            LoggingLevelNames[level] << "] " << buffer << std::endl;
    }
}

void Logger::debug(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::DEBUG) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::DEBUG, format, args);
    va_end(args);
}

void Logger::info(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::INFO) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::INFO, format, args);
    va_end(args);
}

void Logger::warn(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::WARNING) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::WARNING, format, args);
    va_end(args);
}

void Logger::error(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::ERROR) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::ERROR, format, args);
    va_end(args);
}

void Logger::critical(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::CRITICAL) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::CRITICAL, format, args);
    va_end(args);
}

}
