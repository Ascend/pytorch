#ifndef __COMMON_LOGGER_H__
#define __COMMON_LOGGER_H__

#include <cstdint>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/visible.h"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) noexcept {
  os << "{";
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << std::to_string(vec[i]);
  }
  os << "}";
  return os;
}

namespace fxrt {
namespace common {

enum class GLogLevel : int { DEBUG = 0, INFO, WARNING, ERROR, CRITICAL, EXCEPTION = CRITICAL };

constexpr int kVLogModuleRange = 8;

// VLOG uses a single uint64_t mask for fast runtime checks. Therefore valid VLOG levels are bits [0, 63].
// Each source module owns an 8-level range. Keep the explicit range starts stable because users put
// these values in VLOG_v. The *_LAST aliases mark the current last valid level in each range for
// compile-time range-capacity checks.
enum class VLogLevel : uint8_t {
  RUNTIME = 0,
  RUNTIME_DETAIL,
  RUNTIME_MEMORY,
  RUNTIME_PIPELINE,
  RUNTIME_EXECUTOR,
  RUNTIME_BUILDER,
  RUNTIME_CAPTURE,
  RUNTIME_OTHER,
  RUNTIME_LAST = RUNTIME_OTHER,

  OPS = 8,
  OPS_ACLNN,
  OPS_ATB,
  OPS_HCCL,
  OPS_DVM,
  OPS_CPU,
  OPS_CUSTOM,
  OPS_OTHER,
  OPS_LAST = OPS_OTHER,

  HARDWARE = 16,
  HARDWARE_ASCEND,
  HARDWARE_CPU,
  HARDWARE_MEMORY,
  HARDWARE_STREAM,
  HARDWARE_COLLECTIVE,
  HARDWARE_CAPTURE,
  HARDWARE_OTHER,
  HARDWARE_LAST = HARDWARE_OTHER,

  COMMON = 24,
  COMMON_LOADER,
  COMMON_LOGGER,
  COMMON_UTILS,
  COMMON_OTHER_0,
  COMMON_OTHER_1,
  COMMON_OTHER_2,
  COMMON_OTHER,
  COMMON_LAST = COMMON_OTHER,

  CONFIG = 32,
  CONFIG_ASCEND,
  CONFIG_ACLGRAPH,
  CONFIG_OP_PRECISION,
  CONFIG_OTHER_0,
  CONFIG_OTHER_1,
  CONFIG_OTHER_2,
  CONFIG_OTHER,
  CONFIG_LAST = CONFIG_OTHER,

  IR = 40,
  IR_GRAPH,
  IR_TENSOR,
  IR_VALUE,
  IR_SYMBOLIC,
  IR_DTYPE,
  IR_STORAGE,
  IR_OTHER,
  IR_LAST = IR_OTHER,

  OPTIMIZE = 48,
  OPTIMIZE_PASS,
  OPTIMIZE_UD,
  OPTIMIZE_OTHER_0,
  OPTIMIZE_OTHER_1,
  OPTIMIZE_OTHER_2,
  OPTIMIZE_OTHER_3,
  OPTIMIZE_OTHER,
  OPTIMIZE_LAST = OPTIMIZE_OTHER,

  PROFILER = 56,
  PROFILER_TRACE,
  PROFILER_RUNTIME,
  PROFILER_OPS,
  PROFILER_MEMORY,
  PROFILER_OTHER_0,
  FLOW,
  DISP_VLOG_TAGS,
  PROFILER_LAST = DISP_VLOG_TAGS,
};

constexpr uint8_t kMaxVLogLevel = static_cast<uint8_t>(VLogLevel::DISP_VLOG_TAGS);
static_assert(kMaxVLogLevel < 64, "VLogLevel must fit in the 64-bit runtime mask.");
static_assert(
    static_cast<uint8_t>(VLogLevel::RUNTIME_LAST) - static_cast<uint8_t>(VLogLevel::RUNTIME) + 1 <= kVLogModuleRange,
    "VLOG runtime range exceeds its 8-level allocation.");
static_assert(
    static_cast<uint8_t>(VLogLevel::OPS_LAST) - static_cast<uint8_t>(VLogLevel::OPS) + 1 <= kVLogModuleRange,
    "VLOG ops range exceeds its 8-level allocation.");
static_assert(
    static_cast<uint8_t>(VLogLevel::HARDWARE_LAST) - static_cast<uint8_t>(VLogLevel::HARDWARE) + 1 <= kVLogModuleRange,
    "VLOG hardware range exceeds its 8-level allocation.");
static_assert(
    static_cast<uint8_t>(VLogLevel::COMMON_LAST) - static_cast<uint8_t>(VLogLevel::COMMON) + 1 <= kVLogModuleRange,
    "VLOG common range exceeds its 8-level allocation.");
static_assert(
    static_cast<uint8_t>(VLogLevel::CONFIG_LAST) - static_cast<uint8_t>(VLogLevel::CONFIG) + 1 <= kVLogModuleRange,
    "VLOG config range exceeds its 8-level allocation.");
static_assert(
    static_cast<uint8_t>(VLogLevel::IR_LAST) - static_cast<uint8_t>(VLogLevel::IR) + 1 <= kVLogModuleRange,
    "VLOG IR range exceeds its 8-level allocation.");
static_assert(
    static_cast<uint8_t>(VLogLevel::OPTIMIZE_LAST) - static_cast<uint8_t>(VLogLevel::OPTIMIZE) + 1 <= kVLogModuleRange,
    "VLOG optimize range exceeds its 8-level allocation.");
static_assert(
    static_cast<uint8_t>(VLogLevel::PROFILER_LAST) - static_cast<uint8_t>(VLogLevel::PROFILER) + 1 <= kVLogModuleRange,
    "VLOG profiler range exceeds its 8-level allocation.");
constexpr uint64_t kVLogRuntimeMask = 0x00000000000000ffULL;
constexpr uint64_t kVLogOpsMask = 0x000000000000ff00ULL;
constexpr uint64_t kVLogHardwareMask = 0x0000000000ff0000ULL;
constexpr uint64_t kVLogCommonMask = 0x00000000ff000000ULL;
constexpr uint64_t kVLogConfigMask = 0x000000ff00000000ULL;
constexpr uint64_t kVLogIrMask = 0x0000ff0000000000ULL;
constexpr uint64_t kVLogOptimizeMask = 0x00ff000000000000ULL;
constexpr uint64_t kVLogProfilerMask = 0xff00000000000000ULL;

class LogStream {
 public:
  LogStream() = default;
  ~LogStream() = default;

  template <typename T>
  LogStream& operator<<(const T& val) noexcept {
    stream_ << val;
    return *this;
  }

  LogStream& operator<<(std::ostream& func(std::ostream& os)) noexcept {
    stream_ << func;
    return *this;
  }

  const std::ostringstream& Stream() const {
    return stream_;
  }

 private:
  std::ostringstream stream_;
};

class FXRT_EXPORT LogWriter final {
 public:
  LogWriter(const char* file, int line, const char* func, GLogLevel level);
  LogWriter(const char* file, int line, const char* func, VLogLevel vlog_level);
  ~LogWriter() = default;

  void operator<(const LogStream& stream) const;

 private:
  const char* file_;
  int line_;
  const char* func_;
  GLogLevel level_;
  VLogLevel vlog_level_;
  bool is_vlog_;
};

FXRT_EXPORT extern uint64_t g_fxrt_vlog_mask;
FXRT_EXPORT bool IsGlogOn(GLogLevel level);

inline bool IsVlogOn(VLogLevel level) noexcept {
  return ((g_fxrt_vlog_mask >> static_cast<uint8_t>(level)) & 1U) != 0;
}

class ExceptionLogStream final {
 public:
  ExceptionLogStream(const char* file, int line, const char* func, GLogLevel level)
      : writer_(file, line, func, level) {}

  LogStream& Stream() {
    return stream_;
  }

  [[noreturn]] void ThrowNow() {
    writer_ < stream_;
    throw std::runtime_error(stream_.Stream().str());
  }

 private:
  LogWriter writer_;
  LogStream stream_;
};

} // namespace common

// VLOG usage:
//   RT_VLOG(VL_RUNTIME) << "message";
//   if (RT_VLOG_IS_ON(VL_OPS_ACLNN)) { ... expensive debug string construction ...; }
//
// Runtime control is configured by VLOG_v before loading FXRT:
//   export VLOG_v="0,2-3,8-15"  # enable individual levels and inclusive ranges
//   export VLOG_v="63"          # print the reserved VLOG tag usage list
//
// Level 64 is invalid. The implementation stores enabled levels in a uint64_t mask, so the highest
// representable bit is 63.
constexpr common::VLogLevel VL_RUNTIME = common::VLogLevel::RUNTIME;
constexpr common::VLogLevel VL_RUNTIME_DETAIL = common::VLogLevel::RUNTIME_DETAIL;
constexpr common::VLogLevel VL_RUNTIME_MEMORY = common::VLogLevel::RUNTIME_MEMORY;
constexpr common::VLogLevel VL_RUNTIME_PIPELINE = common::VLogLevel::RUNTIME_PIPELINE;
constexpr common::VLogLevel VL_RUNTIME_EXECUTOR = common::VLogLevel::RUNTIME_EXECUTOR;
constexpr common::VLogLevel VL_RUNTIME_BUILDER = common::VLogLevel::RUNTIME_BUILDER;
constexpr common::VLogLevel VL_RUNTIME_CAPTURE = common::VLogLevel::RUNTIME_CAPTURE;
constexpr common::VLogLevel VL_RUNTIME_OTHER = common::VLogLevel::RUNTIME_OTHER;
constexpr common::VLogLevel VL_OPS = common::VLogLevel::OPS;
constexpr common::VLogLevel VL_OPS_ACLNN = common::VLogLevel::OPS_ACLNN;
constexpr common::VLogLevel VL_OPS_ATB = common::VLogLevel::OPS_ATB;
constexpr common::VLogLevel VL_OPS_HCCL = common::VLogLevel::OPS_HCCL;
constexpr common::VLogLevel VL_OPS_DVM = common::VLogLevel::OPS_DVM;
constexpr common::VLogLevel VL_OPS_CPU = common::VLogLevel::OPS_CPU;
constexpr common::VLogLevel VL_OPS_CUSTOM = common::VLogLevel::OPS_CUSTOM;
constexpr common::VLogLevel VL_OPS_OTHER = common::VLogLevel::OPS_OTHER;
constexpr common::VLogLevel VL_HARDWARE = common::VLogLevel::HARDWARE;
constexpr common::VLogLevel VL_HARDWARE_ASCEND = common::VLogLevel::HARDWARE_ASCEND;
constexpr common::VLogLevel VL_HARDWARE_CPU = common::VLogLevel::HARDWARE_CPU;
constexpr common::VLogLevel VL_HARDWARE_MEMORY = common::VLogLevel::HARDWARE_MEMORY;
constexpr common::VLogLevel VL_HARDWARE_STREAM = common::VLogLevel::HARDWARE_STREAM;
constexpr common::VLogLevel VL_HARDWARE_COLLECTIVE = common::VLogLevel::HARDWARE_COLLECTIVE;
constexpr common::VLogLevel VL_HARDWARE_CAPTURE = common::VLogLevel::HARDWARE_CAPTURE;
constexpr common::VLogLevel VL_HARDWARE_OTHER = common::VLogLevel::HARDWARE_OTHER;
constexpr common::VLogLevel VL_COMMON = common::VLogLevel::COMMON;
constexpr common::VLogLevel VL_CONFIG = common::VLogLevel::CONFIG;
constexpr common::VLogLevel VL_IR = common::VLogLevel::IR;
constexpr common::VLogLevel VL_OPTIMIZE = common::VLogLevel::OPTIMIZE;
constexpr common::VLogLevel VL_PROFILER = common::VLogLevel::PROFILER;
constexpr common::VLogLevel VL_FLOW = common::VLogLevel::FLOW;
constexpr common::VLogLevel VL_DISP_VLOG_TAGS = common::VLogLevel::DISP_VLOG_TAGS;
} // namespace fxrt

#define FXRT_LOG_CONCAT_INNER_(a, b) a##b
#define FXRT_LOG_CONCAT_(a, b) FXRT_LOG_CONCAT_INNER_(a, b)

#define FXRT_GLOG_IMPL_(level)       \
  !(::fxrt::common::IsGlogOn(level)) \
      ? static_cast<void>(0)         \
      : ::fxrt::common::LogWriter(__FILE__, __LINE__, __FUNCTION__, level) < ::fxrt::common::LogStream()

#define FXRT_GLOG_EXCEPTION_IMPL_(id, level)                                                  \
  for (::fxrt::common::ExceptionLogStream FXRT_LOG_CONCAT_(_fxrt_glog_exception_stream_, id)( \
           __FILE__, __LINE__, __FUNCTION__, level);                                          \
       ;                                                                                      \
       FXRT_LOG_CONCAT_(_fxrt_glog_exception_stream_, id).ThrowNow())                         \
  FXRT_LOG_CONCAT_(_fxrt_glog_exception_stream_, id).Stream()

#define FXRT_GLOG_DEBUG() FXRT_GLOG_IMPL_(::fxrt::common::GLogLevel::DEBUG)
#define FXRT_GLOG_INFO() FXRT_GLOG_IMPL_(::fxrt::common::GLogLevel::INFO)
#define FXRT_GLOG_WARNING() FXRT_GLOG_IMPL_(::fxrt::common::GLogLevel::WARNING)
#define FXRT_GLOG_ERROR() FXRT_GLOG_IMPL_(::fxrt::common::GLogLevel::ERROR)
#define FXRT_GLOG_CRITICAL() FXRT_GLOG_EXCEPTION_IMPL_(__COUNTER__, ::fxrt::common::GLogLevel::CRITICAL)
#define FXRT_GLOG_EXCEPTION() FXRT_GLOG_EXCEPTION_IMPL_(__COUNTER__, ::fxrt::common::GLogLevel::EXCEPTION)

#define FXRT_GLOG_SELECT_(level) FXRT_GLOG_##level
#define RT_GLOG(level) FXRT_GLOG_SELECT_(level)()

// VLOG is compiled as a short-circuit expression. When the level is disabled by VLOG_v, the stream
// expression on the right is not evaluated. Use RT_VLOG_IS_ON(level) around expensive preparation work.
#define FXRT_VLOG_IMPL_(level)       \
  !(::fxrt::common::IsVlogOn(level)) \
      ? static_cast<void>(0)         \
      : ::fxrt::common::LogWriter(__FILE__, __LINE__, __FUNCTION__, level) < ::fxrt::common::LogStream()

#define RT_VLOG(level) FXRT_VLOG_IMPL_(level)
#define RT_VLOG_IS_ON(level) (::fxrt::common::IsVlogOn(level))

#endif // __COMMON_LOGGER_H__
