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

private:
    void log(LoggingLevel level, const char* format, va_list args);

    LoggingLevel allow_level_ = LoggingLevel::WARNING;
    std::string name_;
    std::string qname_;
};

}

#endif
