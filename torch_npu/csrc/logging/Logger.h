#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <cstdarg>

namespace npu_logging {

enum class LoggingLevel {
    DEBUG = 10,
    INFO = 20,
    WARNING = 30,
    ERROR = 40,
    CRITICAL = 50
};

class Logger {
public:
    Logger() = default;
    Logger(const std::string &name) : name_(name) {};
    ~Logger() = default;

    LoggingLevel getAllowLevel();
    void setAllowLevel(LoggingLevel level);
    void setQName(const std::string& qname);
    std::string getQName();
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

private:
    void log(LoggingLevel level, const std::string& levelStr, const int log_buffer_size, const char* format, va_list args);

    LoggingLevel allow_level_ = LoggingLevel::WARNING;
    std::string name_;
    std::string qname_;
};

} // namespace npu_logging

// public macros for logging normal message
#define TORCH_NPU_LOGD(module, format, ...)                                     \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::DEBUG) {      \
            module->debug(format, ##__VA_ARGS__);                               \
        }                                                                       \
    } while (0);

#define TORCH_NPU_LOGI(module, format, ...)                                     \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::INFO) {       \
            module->info(format, ##__VA_ARGS__);                                \
        }                                                                       \
    } while (0);

#define TORCH_NPU_LOGW(module, format, ...)                                     \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::WARNING) {    \
            module->warn(format, ##__VA_ARGS__);                                \
        }                                                                       \
    } while (0);

#define TORCH_NPU_LOGE(module, format, ...)                                     \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::ERROR) {      \
            module->error(format, ##__VA_ARGS__);                               \
        }                                                                       \
    } while (0);

#define TORCH_NPU_LOGC(module, format, ...)                                     \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::CRITICAL) {   \
            module->critical(format, ##__VA_ARGS__);                            \
        }                                                                       \
    } while (0);


// public macros for logging long message
#define TORCH_NPU_LOGDL(module, format, ...)                                    \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::DEBUG) {      \
            module->long_debug(format, ##__VA_ARGS__);                          \
        }                                                                       \
    } while (0);

#define TORCH_NPU_LOGIL(module, format, ...)                                    \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::INFO) {       \
            module->long_info(format, ##__VA_ARGS__);                           \
        }                                                                       \
    } while (0);

#define TORCH_NPU_LOGWL(module, format, ...)                                    \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::WARNING) {    \
            module->long_warn(format, ##__VA_ARGS__);                           \
        }                                                                       \
    } while (0);

#define TORCH_NPU_LOGEL(module, format, ...)                                    \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::ERROR) {      \
            module->long_error(format, ##__VA_ARGS__);                          \
        }                                                                       \
    } while (0);

#define TORCH_NPU_LOGCL(module, format, ...)                                    \
    do {                                                                        \
        if (module->getAllowLevel() <= npu_logging::LoggingLevel::CRITICAL) {   \
            module->long_critical(format, ##__VA_ARGS__);                       \
        }                                                                       \
    } while (0);

#endif