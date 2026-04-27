#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <cstdarg>
#include <cstdint>

namespace npu_logging {

enum class LoggingLevel {
    DEBUG = 10,
    INFO = 20,
    WARNING = 30,
    ERROR = 40,
    CRITICAL = 50
};

// logging level count
static const int LOGGING_LEVEL_COUNT = 5;

class Logger {
public:
    Logger() = default;
    Logger(const std::string &name) : name_(name) {};
    ~Logger() = default;

    LoggingLevel getAllowLevel();
    void setAllowLevel(LoggingLevel level);
    void setQName(const std::string& qname);
    std::string getQName();

    // For compatibility with older versions, these functions are temporarily retained but may be removed in the future.
    void debug(const char* format, ...);
    void info(const char* format, ...);
    void warn(const char* format, ...);
    void error(const char* format, ...);
    void critical(const char* format, ...);
    void long_debug(const char* format, ...);
    void long_info(const char* format, ...);
    void long_warn(const char* format, ...);
    void long_error(const char* format, ...);
    void long_critical(const char* format, ...);

    // New log functions that include the file name and line number
    void debug(const char* file, uint32_t line, const char* format, ...);
    void info(const char* file, uint32_t line, const char* format, ...);
    void warn(const char* file, uint32_t line, const char* format, ...);
    void error(const char* file, uint32_t line, const char* format, ...);
    void critical(const char* file, uint32_t line, const char* format, ...);
    void long_debug(const char* file, uint32_t line, const char* format, ...);
    void long_info(const char* file, uint32_t line, const char* format, ...);
    void long_warn(const char* file, uint32_t line, const char* format, ...);
    void long_error(const char* file, uint32_t line, const char* format, ...);
    void long_critical(const char* file, uint32_t line, const char* format, ...);

private:
    void log(LoggingLevel level, const int log_buffer_size, const char* file, uint32_t line, const char* format, va_list args);

    LoggingLevel allow_level_ = LoggingLevel::WARNING;
    std::string name_;
    std::string qname_;
};

} // namespace npu_logging

// public macros for logging message
#define TORCH_NPU_LOGD(module, format, ...)                                     \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::DEBUG) {      \
            module->debug(__FILE__, __LINE__, format, ##__VA_ARGS__);           \
        }                                                                       \
    } while (0);

#define TORCH_NPU_LOGI(module, format, ...)                                     \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::INFO) {       \
            module->info(__FILE__, __LINE__, format, ##__VA_ARGS__);            \
        }                                                                       \
    } while (0);

#define TORCH_NPU_LOGW(module, format, ...)                                     \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::WARNING) {    \
            module->warn(__FILE__, __LINE__, format, ##__VA_ARGS__);            \
        }                                                                       \
    } while (0);

#define TORCH_NPU_LOGE(module, format, ...)                                     \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::ERROR) {      \
            module->error(__FILE__, __LINE__, format, ##__VA_ARGS__);           \
        }                                                                       \
    } while (0);

// public macros for logging long message
#define TORCH_NPU_LOGIL(module, format, ...)                                    \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::INFO) {       \
            module->long_info(__FILE__, __LINE__, format, ##__VA_ARGS__);       \
        }                                                                       \
    } while (0);

#endif