#include "common/logger.h"

#include <cctype>
#include <cstdint>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef ENABLE_TORCH_NPU
#include <atomic>
#include <memory>

#include "torch_npu/csrc/logging/LogContext.h"
#include "torch_npu/csrc/logging/Logger.h"
#else
#include <iomanip>
#include <iostream>
#include <thread>
#ifndef _MSC_VER
#include <sys/time.h>
#include <unistd.h>
#endif
#endif

#include "common/common.h"

namespace fxrt {
namespace common {

uint64_t g_fxrt_vlog_mask = 0;

namespace {
constexpr char kSplitLine[] = "----------------------------------------------------\n";
constexpr int kVLogMaxLevel = static_cast<int>(kMaxVLogLevel);

struct VLogRangeDesc {
  VLogLevel begin;
  VLogLevel end;
  const char* module;
};

struct VLogTagDesc {
  VLogLevel level;
  const char* name;
  const char* description;
};

std::once_flag& GetInitFlag() {
  static std::once_flag init_flag;
  return init_flag;
}

bool IsDigit(char ch) {
  return ch >= '0' && ch <= '9';
}

bool ParseNonNegativeInt(const std::string& value, size_t begin, size_t end, int* result) {
  if (begin >= end) {
    return false;
  }

  int64_t parsed_value = 0;
  size_t index = begin;
  while (index < end && IsDigit(value[index])) {
    parsed_value = parsed_value * 10 + (value[index] - '0');
    if (parsed_value > std::numeric_limits<int>::max()) {
      return false;
    }
    ++index;
  }
  if (index != end) {
    return false;
  }
  *result = static_cast<int>(parsed_value);
  return true;
}

std::string Trim(const std::string& value) {
  size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }
  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }
  return value.substr(begin, end - begin);
}

bool ParseVLogLevel(const std::string& value, int* level) {
  const auto token = Trim(value);
  if (token.empty() || !ParseNonNegativeInt(token, 0, token.size(), level)) {
    return false;
  }
  return *level <= kVLogMaxLevel;
}

bool ParseVLogToken(const std::string& token, uint64_t* mask) {
  const auto trimmed = Trim(token);
  if (trimmed.empty()) {
    return false;
  }

  auto dash = trimmed.find('-');
  if (dash == std::string::npos) {
    int level = 0;
    if (!ParseVLogLevel(trimmed, &level)) {
      return false;
    }
    *mask |= (uint64_t{1} << static_cast<uint8_t>(level));
    return true;
  }

  if (dash == 0 || dash + 1 >= trimmed.size() || trimmed.find('-', dash + 1) != std::string::npos) {
    return false;
  }

  int from = 0;
  int to = 0;
  if (!ParseVLogLevel(trimmed.substr(0, dash), &from) || !ParseVLogLevel(trimmed.substr(dash + 1), &to) || from > to) {
    return false;
  }

  for (int level = from; level <= to; ++level) {
    *mask |= (uint64_t{1} << static_cast<uint8_t>(level));
  }
  return true;
}

bool ParseVLogMask(const std::string& value, uint64_t* mask) {
  if (value.empty()) {
    return true;
  }

  // VLOG_v accepts comma-separated levels and inclusive ranges, for example:
  //   VLOG_v="1,2-3,8-15"
  // Parsed levels are stored as bits in g_fxrt_vlog_mask, making IsVlogOn a single shift-and-test.
  uint64_t parsed_mask = 0;
  size_t begin = 0;
  while (begin <= value.size()) {
    const auto comma = value.find(',', begin);
    const auto end = comma == std::string::npos ? value.size() : comma;
    if (!ParseVLogToken(value.substr(begin, end - begin), &parsed_mask)) {
      return false;
    }
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1;
  }

  *mask = parsed_mask;
  return true;
}

bool IsExceptionLevel(GLogLevel level) {
  return static_cast<int>(level) >= static_cast<int>(GLogLevel::CRITICAL);
}

std::string NormalizeFileName(const char* file) {
  if (file == nullptr || file[0] == '\0') {
    return "";
  }

  std::string file_name(file);
  auto pos = file_name.rfind("/fxrt/");
  if (pos != std::string::npos) {
    return file_name.substr(pos + 1);
  }

  pos = file_name.find("fxrt/");
  if (pos != std::string::npos) {
    return file_name.substr(pos);
  }
  return file_name;
}

#ifdef ENABLE_TORCH_NPU
// Name of the torch_npu logger fxrt writes to. It matches the "fxrt" log alias torch_npu registers, so
// TORCH_NPU_LOGS and torch._logging.set_logs set its level together with fxrt's Python loggers.
constexpr char kTorchNpuLoggerName[] = "fxrt";
constexpr char kModulePrefix[] = "[FXRT] ";
// npu_logging::Logger drops a whole record when the formatted message overflows its 4096-byte long-message
// buffer, so a message is written as one record per line, and a line longer than this is split further.
constexpr size_t kMaxRecordBytes = 3072;

std::atomic<bool> g_vlog_suppressed_warned{false};

const std::shared_ptr<npu_logging::Logger>& GetTorchNpuLogger() {
  static const std::shared_ptr<npu_logging::Logger> logger = npu_logging::logging().getLogger(kTorchNpuLoggerName);
  return logger;
}

npu_logging::LoggingLevel ToNpuLoggingLevel(GLogLevel level) {
  switch (level) {
    case GLogLevel::DEBUG:
      return npu_logging::LoggingLevel::DEBUG;
    case GLogLevel::INFO:
      return npu_logging::LoggingLevel::INFO;
    case GLogLevel::WARNING:
      return npu_logging::LoggingLevel::WARNING;
    case GLogLevel::ERROR:
      return npu_logging::LoggingLevel::ERROR;
    case GLogLevel::CRITICAL:
    default:
      return npu_logging::LoggingLevel::CRITICAL;
  }
}

void WriteRecord(
    npu_logging::Logger& logger,
    npu_logging::LoggingLevel level,
    const std::string& file_name,
    int line,
    const std::string& text) {
  const auto line_no = static_cast<uint32_t>(line);
  switch (level) {
    case npu_logging::LoggingLevel::DEBUG:
      logger.long_debug(file_name.c_str(), line_no, "%s", text.c_str());
      break;
    case npu_logging::LoggingLevel::INFO:
      logger.long_info(file_name.c_str(), line_no, "%s", text.c_str());
      break;
    case npu_logging::LoggingLevel::WARNING:
      logger.long_warn(file_name.c_str(), line_no, "%s", text.c_str());
      break;
    case npu_logging::LoggingLevel::ERROR:
      logger.long_error(file_name.c_str(), line_no, "%s", text.c_str());
      break;
    case npu_logging::LoggingLevel::CRITICAL:
    default:
      logger.long_critical(file_name.c_str(), line_no, "%s", text.c_str());
      break;
  }
}

// End of the next record inside [begin, end), never cutting a multi-byte UTF-8 sequence.
size_t FindRecordEnd(const std::string& message, size_t begin, size_t end) {
  if (end - begin <= kMaxRecordBytes) {
    return end;
  }
  size_t cut = begin + kMaxRecordBytes;
  while (cut > begin && (static_cast<unsigned char>(message[cut]) & 0xC0) == 0x80) {
    --cut;
  }
  return cut == begin ? begin + kMaxRecordBytes : cut;
}

void WriteRecords(
    npu_logging::Logger& logger,
    npu_logging::LoggingLevel level,
    const std::string& file_name,
    int line,
    const std::string& prefix,
    const std::string& message) {
  size_t begin = 0;
  do {
    auto end = message.find('\n', begin);
    if (end == std::string::npos) {
      end = message.size();
    }
    // An empty line still produces one record, so multi-line messages keep their shape.
    size_t pos = begin;
    do {
      const size_t record_end = FindRecordEnd(message, pos, end);
      WriteRecord(logger, level, file_name, line, prefix + message.substr(pos, record_end - pos));
      pos = record_end;
    } while (pos < end);
    begin = end + 1;
  } while (begin < message.size());
}

// VLOG_v only selects modules; the records are DEBUG records and follow the fxrt log level. Say so once
// when VLOG_v has picked a record that the level then hides, so an unset level is not mistaken for no output.
void WarnVLogSuppressedOnce(npu_logging::Logger& logger) {
  if (g_vlog_suppressed_warned.exchange(true)) {
    return;
  }
  WriteRecord(
      logger,
      npu_logging::LoggingLevel::WARNING,
      NormalizeFileName(__FILE__),
      __LINE__,
      std::string(kModulePrefix) +
          "VLOG_v is set, but VLOG records are DEBUG records and the fxrt log level is above DEBUG. "
          "Enable them with TORCH_NPU_LOGS=+fxrt or torch._logging.set_logs(modules={\"fxrt\": logging.DEBUG}).");
}
#else
constexpr int kDefaultLogLevel = static_cast<int>(GLogLevel::WARNING);

std::string GetTimeString() {
#if defined(_WIN32) || defined(_WIN64)
  return "";
#else
  constexpr size_t kBufLen = 80;
  constexpr int kWidth = 3;
  constexpr int64_t kUsecToMsec = 1000;
  char buf[kBufLen] = {'\0'};
  timeval cur_time;
  (void)gettimeofday(&cur_time, nullptr);
  tm now;
  (void)localtime_r(&cur_time.tv_sec, &now);
  (void)strftime(buf, kBufLen, "%Y-%m-%d-%H:%M:%S", &now);
  std::stringstream ss;
  ss << buf << "." << std::setfill('0') << std::setw(kWidth) << cur_time.tv_usec / kUsecToMsec << "."
     << std::setfill('0') << std::setw(kWidth) << cur_time.tv_usec % kUsecToMsec;
  return ss.str();
#endif
}

std::string GetLogLevelName(GLogLevel level) {
  static const char* const level_names[] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"};
  return level_names[static_cast<int>(level)];
}
#endif

void PrintVLogTag(const char* file, int line, const char* function, const std::string& msg) {
  LogWriter(file, line, function, VLogLevel::DISP_VLOG_TAGS) < LogStream() << msg;
}

void DispVLogTags() {
  if (!IsVlogOn(VLogLevel::DISP_VLOG_TAGS)) {
    return;
  }

  static constexpr VLogRangeDesc kVLogRangeDescs[] = {
      {VLogLevel::RUNTIME, VLogLevel::RUNTIME_LAST, "runtime"},
      {VLogLevel::OPS, VLogLevel::OPS_LAST, "ops"},
      {VLogLevel::HARDWARE, VLogLevel::HARDWARE_LAST, "hardware"},
      {VLogLevel::COMMON, VLogLevel::COMMON_LAST, "common"},
      {VLogLevel::CONFIG, VLogLevel::CONFIG_LAST, "config"},
      {VLogLevel::IR, VLogLevel::IR_LAST, "IR"},
      {VLogLevel::OPTIMIZE, VLogLevel::OPTIMIZE_LAST, "optimize"},
      {VLogLevel::PROFILER, VLogLevel::PROFILER_OTHER_0, "profiler"},
  };
  static constexpr VLogTagDesc kVLogTagDescs[] = {
      {VLogLevel::RUNTIME, "VL_RUNTIME", "runtime module base log level"},
      {VLogLevel::RUNTIME_DETAIL, "VL_RUNTIME_DETAIL", "runtime detail log level"},
      {VLogLevel::RUNTIME_MEMORY, "VL_RUNTIME_MEMORY", "runtime memory log level"},
      {VLogLevel::RUNTIME_PIPELINE, "VL_RUNTIME_PIPELINE", "runtime pipeline log level"},
      {VLogLevel::RUNTIME_EXECUTOR, "VL_RUNTIME_EXECUTOR", "runtime executor log level"},
      {VLogLevel::RUNTIME_BUILDER, "VL_RUNTIME_BUILDER", "runtime builder log level"},
      {VLogLevel::RUNTIME_CAPTURE, "VL_RUNTIME_CAPTURE", "runtime capture log level"},
      {VLogLevel::RUNTIME_OTHER, "VL_RUNTIME_OTHER", "runtime other log level"},
      {VLogLevel::OPS, "VL_OPS", "ops module base log level"},
      {VLogLevel::OPS_ACLNN, "VL_OPS_ACLNN", "ops aclnn log level"},
      {VLogLevel::OPS_ATB, "VL_OPS_ATB", "ops atb log level"},
      {VLogLevel::OPS_HCCL, "VL_OPS_HCCL", "ops hccl log level"},
      {VLogLevel::OPS_DVM, "VL_OPS_DVM", "ops dvm log level"},
      {VLogLevel::OPS_CPU, "VL_OPS_CPU", "ops cpu log level"},
      {VLogLevel::OPS_CUSTOM, "VL_OPS_CUSTOM", "ops custom log level"},
      {VLogLevel::OPS_OTHER, "VL_OPS_OTHER", "ops other log level"},
      {VLogLevel::HARDWARE, "VL_HARDWARE", "hardware module base log level"},
      {VLogLevel::HARDWARE_ASCEND, "VL_HARDWARE_ASCEND", "hardware ascend log level"},
      {VLogLevel::HARDWARE_CPU, "VL_HARDWARE_CPU", "hardware cpu log level"},
      {VLogLevel::HARDWARE_MEMORY, "VL_HARDWARE_MEMORY", "hardware memory log level"},
      {VLogLevel::HARDWARE_STREAM, "VL_HARDWARE_STREAM", "hardware stream log level"},
      {VLogLevel::HARDWARE_COLLECTIVE, "VL_HARDWARE_COLLECTIVE", "hardware collective log level"},
      {VLogLevel::HARDWARE_CAPTURE, "VL_HARDWARE_CAPTURE", "hardware capture log level"},
      {VLogLevel::HARDWARE_OTHER, "VL_HARDWARE_OTHER", "hardware other log level"},
      {VLogLevel::COMMON, "VL_COMMON", "common module base log level"},
      {VLogLevel::COMMON_LOADER, "VL_COMMON_LOADER", "common loader log level"},
      {VLogLevel::COMMON_LOGGER, "VL_COMMON_LOGGER", "common logger log level"},
      {VLogLevel::COMMON_UTILS, "VL_COMMON_UTILS", "common utils log level"},
      {VLogLevel::COMMON_OTHER_0, "VL_COMMON_OTHER_0", "common reserved log level 0"},
      {VLogLevel::COMMON_OTHER_1, "VL_COMMON_OTHER_1", "common reserved log level 1"},
      {VLogLevel::COMMON_OTHER_2, "VL_COMMON_OTHER_2", "common reserved log level 2"},
      {VLogLevel::COMMON_OTHER, "VL_COMMON_OTHER", "common other log level"},
      {VLogLevel::CONFIG, "VL_CONFIG", "config module base log level"},
      {VLogLevel::CONFIG_ASCEND, "VL_CONFIG_ASCEND", "config ascend log level"},
      {VLogLevel::CONFIG_ACLGRAPH, "VL_CONFIG_ACLGRAPH", "config aclgraph log level"},
      {VLogLevel::CONFIG_OP_PRECISION, "VL_CONFIG_OP_PRECISION", "config op precision log level"},
      {VLogLevel::CONFIG_OTHER_0, "VL_CONFIG_OTHER_0", "config reserved log level 0"},
      {VLogLevel::CONFIG_OTHER_1, "VL_CONFIG_OTHER_1", "config reserved log level 1"},
      {VLogLevel::CONFIG_OTHER_2, "VL_CONFIG_OTHER_2", "config reserved log level 2"},
      {VLogLevel::CONFIG_OTHER, "VL_CONFIG_OTHER", "config other log level"},
      {VLogLevel::IR, "VL_IR", "IR module base log level"},
      {VLogLevel::IR_GRAPH, "VL_IR_GRAPH", "IR graph log level"},
      {VLogLevel::IR_TENSOR, "VL_IR_TENSOR", "IR tensor log level"},
      {VLogLevel::IR_VALUE, "VL_IR_VALUE", "IR value log level"},
      {VLogLevel::IR_SYMBOLIC, "VL_IR_SYMBOLIC", "IR symbolic log level"},
      {VLogLevel::IR_DTYPE, "VL_IR_DTYPE", "IR dtype log level"},
      {VLogLevel::IR_STORAGE, "VL_IR_STORAGE", "IR storage log level"},
      {VLogLevel::IR_OTHER, "VL_IR_OTHER", "IR other log level"},
      {VLogLevel::OPTIMIZE, "VL_OPTIMIZE", "optimize module base log level"},
      {VLogLevel::OPTIMIZE_PASS, "VL_OPTIMIZE_PASS", "optimize pass log level"},
      {VLogLevel::OPTIMIZE_UD, "VL_OPTIMIZE_UD", "optimize UD log level"},
      {VLogLevel::OPTIMIZE_OTHER_0, "VL_OPTIMIZE_OTHER_0", "optimize reserved log level 0"},
      {VLogLevel::OPTIMIZE_OTHER_1, "VL_OPTIMIZE_OTHER_1", "optimize reserved log level 1"},
      {VLogLevel::OPTIMIZE_OTHER_2, "VL_OPTIMIZE_OTHER_2", "optimize reserved log level 2"},
      {VLogLevel::OPTIMIZE_OTHER_3, "VL_OPTIMIZE_OTHER_3", "optimize reserved log level 3"},
      {VLogLevel::OPTIMIZE_OTHER, "VL_OPTIMIZE_OTHER", "optimize other log level"},
      {VLogLevel::PROFILER, "VL_PROFILER", "profiler module base log level"},
      {VLogLevel::PROFILER_TRACE, "VL_PROFILER_TRACE", "profiler trace log level"},
      {VLogLevel::PROFILER_RUNTIME, "VL_PROFILER_RUNTIME", "profiler runtime log level"},
      {VLogLevel::PROFILER_OPS, "VL_PROFILER_OPS", "profiler ops log level"},
      {VLogLevel::PROFILER_MEMORY, "VL_PROFILER_MEMORY", "profiler memory log level"},
      {VLogLevel::PROFILER_OTHER_0, "VL_PROFILER_OTHER_0", "profiler reserved log level 0"},
      {VLogLevel::FLOW, "VL_FLOW", "flow vlog level"},
      {VLogLevel::DISP_VLOG_TAGS, "VL_DISP_VLOG_TAGS", "log level for printing vlog tags already been used"},
  };

  // This prints the reserved VLOG tag map itself. It is useful when the user sets VLOG_v to 63
  // to discover which ranges are available in this build.
  PrintVLogTag(
      __FILE__,
      __LINE__,
      __FUNCTION__,
      "VLOG usage: export VLOG_v=\"0,2-3,8-15\" to enable individual levels and inclusive ranges.");
  PrintVLogTag(__FILE__, __LINE__, __FUNCTION__, "VLOG module ranges:");
  for (const auto& range : kVLogRangeDescs) {
    std::stringstream ss;
    ss << static_cast<int>(range.begin) << "-" << static_cast<int>(range.end) << ": " << range.module
       << " module vlog levels";
    PrintVLogTag(__FILE__, __LINE__, __FUNCTION__, ss.str());
  }
  PrintVLogTag(__FILE__, __LINE__, __FUNCTION__, "VLOG tags:");
  for (const auto& tag : kVLogTagDescs) {
    std::stringstream ss;
    ss << static_cast<int>(tag.level) << ": " << tag.name << " - " << tag.description;
    PrintVLogTag(__FILE__, __LINE__, __FUNCTION__, ss.str());
  }
}

void InitLogConfig() {
  const auto vlog_v = GetEnv("VLOG_v");
  if (!vlog_v.empty()) {
    uint64_t vlog_mask = 0;
    if (ParseVLogMask(vlog_v, &vlog_mask)) {
      g_fxrt_vlog_mask = vlog_mask;
    } else {
      g_fxrt_vlog_mask = 0;
      RT_GLOG(WARNING) << "Value of environment var VLOG_v is invalid: " << vlog_v;
    }
  }
  DispVLogTags();
}

void EnsureLogInitialized() {
  std::call_once(GetInitFlag(), InitLogConfig);
}

struct LogInitializer {
  LogInitializer() {
    EnsureLogInitialized();
  }
};

LogInitializer g_log_initializer;

std::string BuildExceptionMessage(const std::string& message, const std::string& file_name, int line) {
  std::stringstream ss;
  ss << message;
  if (!file_name.empty()) {
    ss << "\n" << kSplitLine << "- C++ Call Stack: (For framework developers) \n" << kSplitLine;
    ss << file_name << "(" << line << ").\n\n";
  }
  return ss.str();
}

void OutputLogMessage(
    const char* file,
    int line,
    const char* func,
    GLogLevel level,
    VLogLevel vlog_level,
    bool is_vlog,
    const std::string& message) {
  const std::string file_name = NormalizeFileName(file);
  const auto vlog_level_value = static_cast<int>(vlog_level);
#ifdef ENABLE_TORCH_NPU
  static_cast<void>(func);
  auto& logger = *GetTorchNpuLogger();
  const auto npu_level = is_vlog ? npu_logging::LoggingLevel::DEBUG : ToNpuLoggingLevel(level);
  if (logger.getAllowLevel() > npu_level) {
    if (is_vlog) {
      WarnVLogSuppressedOnce(logger);
    }
    return;
  }
  std::string prefix = kModulePrefix;
  if (is_vlog) {
    prefix += "[V" + std::to_string(vlog_level_value) + "] ";
  }
  WriteRecords(logger, npu_level, file_name, line, prefix, message);
#else
  std::cerr << "[" << (is_vlog ? "V" + std::to_string(vlog_level_value) : GetLogLevelName(level)) << "] "
            << GetTimeString() << " [FXRT] [pid:" << getpid() << ", thread id:" << std::hex
            << std::this_thread::get_id() << std::dec << " " << file_name << ":" << line << " " << func << "] "
            << message << std::endl;
#endif
}
} // namespace

LogWriter::LogWriter(const char* file, int line, const char* func, GLogLevel level)
    : file_(file), line_(line), func_(func), level_(level), vlog_level_(VLogLevel::DISP_VLOG_TAGS), is_vlog_(false) {}

LogWriter::LogWriter(const char* file, int line, const char* func, VLogLevel vlog_level)
    : file_(file), line_(line), func_(func), level_(GLogLevel::INFO), vlog_level_(vlog_level), is_vlog_(true) {}

void LogWriter::operator<(const LogStream& stream) const {
  const bool should_throw = IsExceptionLevel(level_);
  std::string message;
  try {
    message = stream.Stream().str();
    if (should_throw) {
      message = BuildExceptionMessage(message, NormalizeFileName(file_), line_);
    }
    OutputLogMessage(file_, line_, func_, level_, vlog_level_, is_vlog_, message);
  } catch (...) {
    if (!should_throw) {
      return;
    }
    message = BuildExceptionMessage(
        "Exception occurred while formatting or writing log message.", NormalizeFileName(file_), line_);
  }
  if (should_throw) {
    throw std::runtime_error(message);
  }
}

bool IsGlogOn(GLogLevel level) {
#ifdef ENABLE_TORCH_NPU
  return GetTorchNpuLogger()->getAllowLevel() <= ToNpuLoggingLevel(level);
#else
  return static_cast<int>(level) >= kDefaultLogLevel;
#endif
}
} // namespace common
} // namespace fxrt
