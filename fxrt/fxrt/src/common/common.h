#ifndef __COMMON_COMMON_H__
#define __COMMON_COMMON_H__

#include <algorithm>
#include <codecvt>
#include <cstring>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <limits>
#include <cstdint>

#include "common/logger.h"

#define ENDL '\n'

#define TO_STR(s) #s

#ifndef CHECK_IF_NULL
#define CHECK_IF_NULL(a)                                                 \
  if (a == nullptr) {                                                    \
    RT_GLOG(EXCEPTION) << '\'' << TO_STR(a) << "\' should not be null."; \
  }
#endif

#define CHECK_IF_FAIL(a)                                                      \
  if (!(a)) {                                                                 \
    RT_GLOG(EXCEPTION) << '\'' << TO_STR(a) << "\' is not true. check fail."; \
  }

#define CHECK_IF_FAIL_MSG(a, msg)  \
  do {                             \
    if (FXRT_UNLIKELY(!(a))) {     \
      RT_GLOG(EXCEPTION) << (msg); \
    }                              \
  } while (0)

#define EVER \
  ;          \
  ; // NOLINT(whitespace/semicolon)

#define DISABLE_COPY_AND_ASSIGN(ClassType) \
  ClassType(const ClassType&) = delete;    \
  ClassType& operator=(const ClassType&) = delete;

inline int8_t Uint32ToInt8(uint32_t u) {
  if (u > static_cast<uint32_t>((std::numeric_limits<int8_t>::max)())) {
    RT_GLOG(EXCEPTION) << "The uint32_t value(" << u << ") exceeds the maximum value of int8_t.";
  }
  return static_cast<int8_t>(u);
}

inline uint32_t LongToUint(int64_t u) {
  if (u < 0) {
    RT_GLOG(EXCEPTION) << "The int64_t value(" << u << ") is less than 0.";
  }
  if (u > static_cast<int64_t>((std::numeric_limits<uint32_t>::max)())) {
    RT_GLOG(EXCEPTION) << "The int64_t value(" << u << ") exceeds the maximum value of uint32_t.";
  }
  return static_cast<uint32_t>(u);
}

inline size_t FloatToSize(float u) {
  if (u < 0) {
    RT_GLOG(EXCEPTION) << "The float value(" << u << ") is less than 0.";
  }

  if (u > static_cast<float>((std::numeric_limits<size_t>::max)())) {
    RT_GLOG(EXCEPTION) << "The float value(" << u << ") exceeds the maximum value of size_t.";
  }
  return static_cast<size_t>(u);
}
inline float IntToFloat(int32_t v) {
  return static_cast<float>(v);
}

inline size_t LongToSize(int64_t u) {
  if (u < 0) {
    RT_GLOG(EXCEPTION) << "The int64_t value(" << u << ") is less than 0.";
  }
  return static_cast<size_t>(u);
}

inline int FloatToInt(float u) {
  if (u > static_cast<float>((std::numeric_limits<int>::max)())) {
    RT_GLOG(EXCEPTION) << "The float value(" << u << ") exceeds the maximum value of int.";
  }
  return static_cast<int>(u);
}

inline int FloatToLong(float u) {
  if (u > static_cast<float>((std::numeric_limits<int64_t>::max)())) {
    RT_GLOG(EXCEPTION) << "The float value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

inline int64_t DoubleToLong(double u) {
  if (u > static_cast<double>((std::numeric_limits<int64_t>::max)())) {
    RT_GLOG(EXCEPTION) << "The double value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

inline float SizeToFloat(size_t v) {
  return static_cast<float>(v);
}

inline uint64_t SizeToUlong(size_t u) {
  return static_cast<uint64_t>(u);
}

inline int SizeToInt(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    RT_GLOG(EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of int.";
  }
  return static_cast<int>(u);
}

inline uint32_t SizeToUint(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<uint32_t>::max)())) {
    RT_GLOG(EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of uint32_t.";
  }
  return static_cast<uint32_t>(u);
}

inline int64_t SizeToLong(size_t u) {
  if (u > static_cast<size_t>((std::numeric_limits<int64_t>::max)())) {
    RT_GLOG(EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

inline double LongToDouble(int64_t v) {
  return static_cast<double>(v);
}

inline float LongToFloat(int64_t v) {
  return static_cast<float>(v);
}

inline double FloatToDouble(float v) {
  return static_cast<double>(v);
}

inline uint32_t IntToUint(int32_t u) {
  if (u < 0) {
    RT_GLOG(EXCEPTION) << "The int32_t value(" << u << ") is less than 0.";
  }
  return static_cast<uint32_t>(u);
}

inline int32_t UintToInt(uint32_t u) {
  if (u > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)())) {
    RT_GLOG(EXCEPTION) << "The uint32_t value(" << u << ") exceeds the maximum value of int32_t.";
  }
  return static_cast<int32_t>(u);
}

inline uint64_t LongToUlong(int64_t u) {
  if (u < 0) {
    RT_GLOG(EXCEPTION) << "The int64_t value(" << u << ") is less than 0.";
  }
  return static_cast<uint64_t>(u);
}

inline int32_t LongToInt(int64_t u) {
  if (u > static_cast<int64_t>((std::numeric_limits<int32_t>::max)())) {
    RT_GLOG(EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of int.";
  }
  return static_cast<int32_t>(u);
}

inline int64_t IntToLong(int32_t v) {
  return static_cast<int64_t>(v);
}

inline int64_t UlongToLong(uint64_t u) {
  if (u > static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
    RT_GLOG(EXCEPTION) << "The uint64_t value(" << u << ") exceeds the maximum value of int64_t.";
  }
  return static_cast<int64_t>(u);
}

inline unsigned int UlongToUint(uint64_t u) {
  if (u > static_cast<uint64_t>((std::numeric_limits<unsigned int>::max)())) {
    RT_GLOG(EXCEPTION) << "The size_t value(" << u << ") exceeds the maximum value of unsigned int.";
  }
  return static_cast<unsigned int>(u);
}

inline uint8_t* AddressOffset(void* address, size_t offset) {
  CHECK_IF_NULL(address);
  return static_cast<uint8_t*>(address) + offset;
}

inline size_t CalAddressOffset(void* dst_address, void* ori_address) {
  CHECK_IF_NULL(dst_address);
  CHECK_IF_NULL(ori_address);
  return static_cast<uint8_t*>(dst_address) - static_cast<uint8_t*>(ori_address);
}

static inline void CompileMessage(const std::string& filename, const int line, const int col, const std::string& msg) {
  std::cout << filename << ':' << line << ':' << (col + 1) << ": " << msg << '\n';
}

static inline void CompileMessage(const std::string& line_info, const std::string& msg) {
  std::cout << line_info << ": " << msg << '\n';
}

// Skip blank, return blank count.
static inline size_t SkipWhiteSpace(const char* str) {
  size_t pos = 0;
  while (str[pos] != '\0') {
    char c = str[pos];
    switch (c) {
      case ' ':
      case '\t':
        ++pos;
        break;
      default:
        return pos;
    }
  }
  return pos;
}

#ifndef FXRT_UNLIKELY
#ifdef _MSC_VER
#define FXRT_UNLIKELY(x) (x)
#define FXRT_LIKELY(x) (x)
#else
#define FXRT_LIKELY(x) __builtin_expect(!!(x), 1)
#define FXRT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#endif

template <typename T>
int FindNameIndex(const char* str, T* table, size_t tableSize) {
  const auto strLen = strlen(str);
  for (size_t i = 0; i < tableSize; ++i) {
    T& element = table[i];
    const auto elementNameLen = strlen(element.name);
    if (elementNameLen <= strLen && strncmp(str, element.name, elementNameLen) == 0) {
      return i;
    }
  }
  return -1;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
inline std::wstring StringToWString(const std::string& str) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.from_bytes(str);
}

inline std::string WStringToString(const std::wstring& wstr) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(wstr);
}
#pragma GCC diagnostic pop

inline std::string ConvertEscapeString(const std::string& str) {
  std::stringstream ss;
  std::string::const_iterator it = str.cbegin();
  while (it != str.cend()) {
    char c = *it++;
    // https://en.cppreference.com/w/cpp/language/escape
    switch (c) {
      case '\'':
        ss << "\'";
        break;
      case '\"':
        ss << "\\\"";
        break;
      case '\?':
        ss << "\\?";
        break;
      case '\\':
        ss << "\\\\";
        break;
      case '\a':
        ss << "\\a";
        break;
      case '\b':
        ss << "\\b";
        break;
      case '\f':
        ss << "\\f";
        break;
      case '\n':
        ss << "\\n";
        break;
      case '\r':
        ss << "\\r";
        break;
      case '\t':
        ss << "\\t";
        break;
      case '\v':
        ss << "\\v";
        break;
      default:
        ss << c;
        break;
    }
  }
  return ss.str();
}

static inline std::string GetEnv(const char* env) {
  if (env == nullptr) {
    return std::string();
  }
  const char* value = std::getenv(env);
  if (value == nullptr) {
    return std::string();
  }

  return std::string(value);
}

static inline bool IsPositiveInteger(const std::string& str) {
  if (str.empty()) {
    return false;
  }
  for (char c : str) {
    if (!std::isdigit(c)) {
      return false;
    }
  }
  return str.front() != '0';
}

#endif // __COMMON_COMMON_H__
