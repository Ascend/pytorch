#pragma once
#include <stdint.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>

#include <ATen/core/ivalue.h>
#include <ATen/record_function.h>

namespace torch_npu {
namespace toolkit {
namespace profiler {

template<typename T>
std::string to_string(T value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

template<typename T>
void encodeFixedData(const std::vector<T> &datas, std::vector<uint8_t> &result) {
  for (auto data : datas) {
    for (size_t i = 0; i < sizeof(T); ++i) {
      result.push_back((static_cast<size_t>(data) >> (i * 8)) & 0xff);
    }
  }
}

inline void encodeStrData(uint16_t type, const std::string &data, std::vector<uint8_t> &result) {
  for (size_t i = 0; i < sizeof(uint16_t); ++i) {
    result.push_back((type >> (i * 8)) & 0xff);
  }
  uint32_t length = data.size();
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    result.push_back((length >> (i * 8)) & 0xff);
  }
  for (const auto &c : data) {
    result.push_back(c);
  }
}

inline void encodeStrArrayData(uint16_t type, const std::vector<std::string> &datas, std::vector<uint8_t> &result) {
  std::string rst;
  for (auto str : datas) {
    rst += (str + ";");
  }
  if (!rst.empty()) {
    rst.pop_back();
  }
  encodeStrData(type, rst, result);
}

inline void encodeMapData(uint16_t type, const std::unordered_map<std::string, c10::IValue> &datas, std::vector<uint8_t> &result) {
  std::string rst;
  for (auto &entry : datas) {
    rst += entry.first + ":" + to_string(entry.second) + ";";
  }
  if (!rst.empty()) {
    rst.pop_back();
  }
  encodeStrData(type, rst, result);
}

template<typename T>
void encode2DIntegerMatrixDatas(uint16_t type, std::vector<std::vector<T>> &datas, std::vector<uint8_t> &result) {
  std::string rst;
  for (auto tensor : datas) {
    std::stringstream ss;
    copy(tensor.begin(), tensor.end(), std::ostream_iterator<T>(ss, ","));
    std::string str = ss.str();
    if (!str.empty()) {
      str.pop_back();
    }
    rst += (str + ";");
  }
  if (!rst.empty()) {
    rst.pop_back();
  }
  encodeStrData(type, rst, result);
}

struct BaseReportData {
  int32_t device_id{0};
  std::string tag;
  BaseReportData(int32_t device_id, std::string tag)
    : device_id(device_id),
      tag(std::move(tag)) {}
  virtual ~BaseReportData() = default;
  virtual std::vector<uint8_t> encode() = 0;
};

enum class OpRangeDataType {
  OP_RANGE_DATA = 1,
  IS_ASYNC = 2,
  NAME = 3,
  INPUT_DTYPES = 4,
  INPUT_SHAPE = 5,
  STACK = 6,
  MODULE_HIERARCHY = 7,
  EXTRA_ARGS = 8,
  RESERVED = 30,
};

struct OpRangeData : BaseReportData{
    int64_t start_ns{0};
    int64_t end_ns{0};
    int64_t sequence_number{0};
    uint64_t process_id{0};
    uint64_t start_thread_id{0};
    uint64_t end_thread_id{0};
    uint64_t forward_thread_id{0};
    bool is_async{false};
    std::string name;
    std::vector<std::string> input_dtypes;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::string> stack;
    std::vector<std::string> module_hierarchy;
    std::unordered_map<std::string, c10::IValue> extra_args;
    OpRangeData(int64_t start_ns,
        int64_t end_ns,
        int64_t sequence_number,
        uint64_t process_id,
        uint64_t start_thread_id,
        uint64_t end_thread_id,
        uint64_t forward_thread_id,
        bool is_async,
        std::string name)
        : BaseReportData(0, "torch.op_range"),
          start_ns(start_ns),
          end_ns(end_ns),
          sequence_number(sequence_number),
          process_id(process_id),
          start_thread_id(start_thread_id),
          end_thread_id(end_thread_id),
          forward_thread_id(forward_thread_id),
          is_async(is_async),
          name(std::move(name)) {}
    std::vector<uint8_t> encode();
};

enum class OpMarkDataType {
  OP_MARK_DATA = 1,
  NAME = 2,
};

struct OpMarkData : BaseReportData {
    int64_t time_ns{0};
    uint64_t category{0};
    uint64_t correlation_id{0};
    uint64_t thread_id{0};
    uint64_t process_id{0};
    std::string name;
    OpMarkData(
        int64_t time_ns,
        uint64_t category,
        uint64_t correlation_id,
        uint64_t thread_id,
        uint64_t process_id,
        const std::string &name)
        : BaseReportData(0, "torch.op_mark"),
          time_ns(time_ns),
          category(category),
          correlation_id(correlation_id),
          thread_id(thread_id),
          process_id(process_id),
          name(name) {}
    std::vector<uint8_t> encode();
};

enum class MemoryDataType {
  MEMORY_DATA = 1,
};

struct MemoryData : BaseReportData {
    int64_t ptr{0};
    int64_t time_ns{0};
    int64_t alloc_size{0};
    int64_t total_allocated{0};
    int64_t total_reserved{0};
    int64_t total_active{0};
    int64_t stream_ptr{0};
    int8_t device_type{0};
    uint8_t device_index{0};
    uint8_t data_type{0};
    uint64_t thread_id{0};
    uint64_t process_id{0};
    MemoryData(
        int64_t ptr,
        int64_t time_ns,
        int64_t alloc_size,
        int64_t total_allocated,
        int64_t total_reserved,
        int64_t total_active,
        int64_t stream_ptr,
        int8_t device_type,
        uint8_t device_index,
        uint8_t data_type,
        uint64_t thread_id,
        uint64_t process_id)
        : BaseReportData(0, "torch.memory_usage"),
          ptr(ptr),
          time_ns(time_ns),
          alloc_size(alloc_size),
          total_allocated(total_allocated),
          total_reserved(total_reserved),
          total_active(total_active),
          stream_ptr(stream_ptr),
          device_type(device_type),
          device_index(device_index),
          data_type(data_type),
          thread_id(thread_id),
          process_id(process_id) {}
    std::vector<uint8_t> encode();
};
} // profiler
} // toolkit
} // torch_npu
