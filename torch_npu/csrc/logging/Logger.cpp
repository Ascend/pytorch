#include <iostream>
#include <cstdio>
#include <ctime>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <sys/syscall.h>
#include "torch_npu/csrc/logging/Logger.h"
#include "torch_npu/csrc/logging/LogContext.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace npu_logging {
static const int BASE_PRINT_LIMIT = 1024;
static const int LONG_PRINT_LIMIT = 4096;

void Logger::setAllowLevel(LoggingLevel level)
{
    allow_level_ = level;
}

LoggingLevel Logger::getAllowLevel()
{
    return allow_level_;
}

void Logger::setQName(const std::string& qname)
{
    qname_ = qname;
}

std::string Logger::getQName()
{
    return qname_;
}

void Logger::log(LoggingLevel level, const std::string& levelStr, const int log_buffer_size, const char* format, va_list args)
{
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
    std::ostringstream oss;
    if (rank != -1) {
        oss << "[rank:" << rank << "]:";
    }
    // Keep 3 decimal places for milliseconds.
    oss << "[" << getpid() << "] [" << timeBuffer << ":" << std::setfill('0') << std::setw(3) << nowMs << "] "
        << name_ << ": [" << levelStr << "] [" << syscall(SYS_gettid) << "] " << buffer << std::endl;
    std::string s = oss.str();
    if (!npu_logging::should_log(s)) {
        return;
    }
    std::cerr.write(s.c_str(), s.size());
    std::cerr.flush();

    // plog
    if (level == LoggingLevel::DEBUG) {
        ASCEND_LOGD("[%s] %s", name_.c_str(), buffer);
    } else if (level == LoggingLevel::INFO) {
        ASCEND_LOGI("[%s] %s", name_.c_str(), buffer);
    } else if (level == LoggingLevel::WARNING) {
        ASCEND_LOGW("[%s] %s", name_.c_str(), buffer);
    } else {
        ASCEND_LOGE("[%s] %s", name_.c_str(), buffer);
    }
}

void Logger::debug(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::DEBUG) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::DEBUG, "DEBUG", BASE_PRINT_LIMIT, format, args);
    va_end(args);
}

void Logger::info(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::INFO) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::INFO, "INFO", BASE_PRINT_LIMIT, format, args);
    va_end(args);
}

void Logger::warn(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::WARNING) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::WARNING, "WARNING", BASE_PRINT_LIMIT, format, args);
    va_end(args);
}

void Logger::error(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::ERROR) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::ERROR, "ERROR", BASE_PRINT_LIMIT, format, args);
    va_end(args);
}

void Logger::critical(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::CRITICAL) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::CRITICAL, "CRITICAL", BASE_PRINT_LIMIT, format, args);
    va_end(args);
}

void Logger::long_debug(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::DEBUG) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::DEBUG, "DEBUG", LONG_PRINT_LIMIT, format, args);
    va_end(args);
}

void Logger::long_info(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::INFO) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::INFO, "INFO", LONG_PRINT_LIMIT, format, args);
    va_end(args);
}

void Logger::long_warn(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::WARNING) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::WARNING, "WARNING", LONG_PRINT_LIMIT, format, args);
    va_end(args);
}

void Logger::long_error(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::ERROR) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::ERROR, "ERROR", LONG_PRINT_LIMIT, format, args);
    va_end(args);
}

void Logger::long_critical(const char* format, ...)
{
    if (allow_level_ > LoggingLevel::CRITICAL) {
        return;
    }
    va_list args;
    va_start(args, format);
    log(LoggingLevel::CRITICAL, "CRITICAL", LONG_PRINT_LIMIT, format, args);
    va_end(args);
}

}
