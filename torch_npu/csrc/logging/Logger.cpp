#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <ctime>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <array>
#include <sys/syscall.h>
#include "torch_npu/csrc/logging/Logger.h"
#include "torch_npu/csrc/logging/LogContext.h"
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace npu_logging {
static const int BASE_PRINT_LIMIT = 1024;
static const int LONG_PRINT_LIMIT = 4096;
static const int PREFIX_MAX_LEN = 256;

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

void Logger::log(LoggingLevel level, const int log_buffer_size, const char* file, uint32_t line, const char* format, va_list args)
{
    char* rankId_val = std::getenv("RANK");
    int64_t rank = (rankId_val != nullptr) ? strtol(rankId_val, nullptr, 10) : -1;
    // Get the mapping character for log level
    // levelChars index mapping: DEBUG(10)->0, INFO(20)->1, WARNING(30)->2, ERROR(40)->3, CRITICAL(50)->4
    // index calculation: (int)level / 10 - 1
    static const std::array<char, LOGGING_LEVEL_COUNT> levelChars = {'V', 'I', 'W', 'E', 'F'};
    char levelChar = levelChars[static_cast<int>(level) / 10 - 1];
    struct timespec ts = {0};
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm = {0};
    localtime_r(&ts.tv_sec, &tm);
    // Convert nanosecond to microsecond (keep 6 digits)
    long microsecond = ts.tv_nsec / 1000;
    std::string rank_str = (rank != -1) ? "[rank:" + std::to_string(rank) + "] " : "";

    char prefix[PREFIX_MAX_LEN] = {0};
    int prefix_len = 0;
    if (file == nullptr || line <= 0) {
        prefix_len = snprintf(prefix, PREFIX_MAX_LEN, "%c%02d%02d %02d:%02d:%02d.%06ld %d] %s", levelChar,
            tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, microsecond, getpid(), rank_str.c_str());
    } else {
        prefix_len = snprintf(prefix, PREFIX_MAX_LEN, "%c%02d%02d %02d:%02d:%02d.%06ld %d %s:%d] %s", levelChar,
            tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, microsecond, getpid(), file, line, rank_str.c_str());
    }

    if (prefix_len < 0 || prefix_len >= PREFIX_MAX_LEN) {
        TORCH_NPU_WARN_ONCE("Failed to generate log prefix.");
        return;
    }

    char buffer[log_buffer_size] = {0};
    int buffer_len = vsnprintf(buffer, log_buffer_size, format, args);
    if (buffer_len < 0 || buffer_len >= log_buffer_size) {
        TORCH_NPU_WARN_ONCE("Failed to generate log message.");
        return;
    }

    std::ostringstream oss;
    oss << prefix << buffer << std::endl;
    std::string s = oss.str();

    if (!npu_logging::should_log(s)) {
        return;
    }

    std::cerr.write(s.c_str(), s.size());
    std::cerr.flush();
}

// Define the basic logging function macro
#define DEFINE_LOG_FUNCTION(func_name, level_enum, buffer_size)                      \
    void Logger::func_name(const char* file, uint32_t line, const char* format, ...) \
    {                                                                                \
        if (allow_level_ > level_enum) {                                             \
            return;                                                                  \
        }                                                                            \
        va_list args;                                                                \
        va_start(args, format);                                                      \
        log(level_enum, buffer_size, file, line, format, args);                      \
        va_end(args);                                                                \
    }                                                                                \
    void Logger::func_name(const char* format, ...)                                  \
    {                                                                                \
        if (allow_level_ > level_enum) {                                             \
            return;                                                                  \
        }                                                                            \
        va_list args;                                                                \
        va_start(args, format);                                                      \
        log(level_enum, buffer_size, nullptr, 0, format, args);                      \
        va_end(args);                                                                \
    }

// Define short format logging functions (using BASE_PRINT_LIMIT buffer size)
DEFINE_LOG_FUNCTION(debug,    LoggingLevel::DEBUG,     BASE_PRINT_LIMIT)
DEFINE_LOG_FUNCTION(info,     LoggingLevel::INFO,      BASE_PRINT_LIMIT)
DEFINE_LOG_FUNCTION(warn,     LoggingLevel::WARNING,   BASE_PRINT_LIMIT)
DEFINE_LOG_FUNCTION(error,    LoggingLevel::ERROR,     BASE_PRINT_LIMIT)
DEFINE_LOG_FUNCTION(critical, LoggingLevel::CRITICAL,  BASE_PRINT_LIMIT)

// Define long format logging functions (using LONG_PRINT_LIMIT buffer size)
DEFINE_LOG_FUNCTION(long_debug,    LoggingLevel::DEBUG,     LONG_PRINT_LIMIT)
DEFINE_LOG_FUNCTION(long_info,     LoggingLevel::INFO,      LONG_PRINT_LIMIT)
DEFINE_LOG_FUNCTION(long_warn,     LoggingLevel::WARNING,   LONG_PRINT_LIMIT)
DEFINE_LOG_FUNCTION(long_error,    LoggingLevel::ERROR,     LONG_PRINT_LIMIT)
DEFINE_LOG_FUNCTION(long_critical, LoggingLevel::CRITICAL,  LONG_PRINT_LIMIT)
}
