#pragma once
#include <stdint.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>

#include <ATen/core/ivalue.h>

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
      result.push_back((data >> (i * 8)) & 0xff);
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
  BaseReportData() {}
  BaseReportData(int32_t device_id, std::string tag) {
    this->device_id = device_id;
    this->tag = tag;
  }
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
  OpRangeData(int32_t device_id, std::string tag) : BaseReportData(device_id, tag) {}
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
  OpMarkData(int32_t device_id, std::string tag,
             int64_t time_ns, uint64_t category,
             uint64_t correlation_id,
             uint64_t thread_id, uint64_t process_id,
             std::string name) {
    this->device_id = device_id;
    this->tag = tag;
    this->time_ns = time_ns;
    this->category = category;
    this->correlation_id = correlation_id;
    this->thread_id = thread_id;
    this->process_id = process_id;
    this->name = name;
  }
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
  int8_t device_type{0};
  uint8_t device_index{0};
  uint64_t thread_id{0};
  uint64_t process_id{0};
  MemoryData(int32_t device_id, std::string tag,
             int64_t ptr, int64_t time_ns, int64_t alloc_size,
             int64_t total_allocated, int64_t total_reserved,
             int8_t device_type, uint8_t device_index,
             uint64_t thread_id, uint64_t process_id) {
    this->device_id = device_id;
    this->tag = tag;
    this->ptr = ptr;
    this->time_ns = time_ns;
    this->alloc_size = alloc_size;
    this->total_allocated = total_allocated;
    this->total_reserved = total_reserved;
    this->device_type = device_type;
    this->device_index = device_index;
    this->thread_id = thread_id;
    this->process_id = process_id;
  }
  std::vector<uint8_t> encode();
};
} // profiler
} // toolkit
} // torch_npu
