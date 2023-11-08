#include <unistd.h>

#include <vector>
#include <map>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

#include "torch_npu/csrc/toolkit/profiler/inc/data_dumper.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"

namespace torch_npu {
namespace toolkit {
namespace profiler {
DataDumper::DataDumper()
    : path_(""),
      start_(false),
      init_(false),
      entry_nums_(0) {}

DataDumper::~DataDumper() {
  UnInit();
}

void DataDumper::Init(const std::string &path, size_t capacity = kDefaultRingBuffer) {
  path_ = path;
  data_chunk_buf_.Init(capacity);
  init_.store(true);
  entry_nums_ = 0;
}

void DataDumper::UnInit() {
  if (init_.load()) {
    data_chunk_buf_.UnInit();
    init_.store(false);
    start_.store(false);
    entry_nums_ = 0;
  }
}

void DataDumper::Start() {
  if (!init_.load() || !Utils::CreateDir(path_)) {
    return;
  }
  if (Thread::Start() != 0) {
    return;
  }
  start_.store(true);
}

void DataDumper::Stop() {
  if (start_.load() == true) {
    start_.store(false);
    Thread::Stop();
    Flush();
  }
}

void DataDumper::GatherAndDumpData() {
  std::map<std::string, std::string> dataMap;
  DataClassifyGather(dataMap);
  if (dataMap.size() > 0) {
    Dump(dataMap);
  }
}

void DataDumper::Run() {
  for (;;) {
    if (!start_.load()) {
      break;
    }
    std::unique_lock<std::mutex> lk(cv_buffer_empty_mtx_);
    cv_buffer_empty_.wait_for(lk, std::chrono::microseconds(kMaxWaitTimeUs),
      [this] {
        return this->data_chunk_buf_.Size() != 0;
      });
    GatherAndDumpData();
  }
}

void DataDumper::Flush() {
  while (data_chunk_buf_.Size() != 0) {
    GatherAndDumpData();
  }
}

void DataDumper::Report(std::unique_ptr<BaseReportData> data) {
  if (!start_.load() || data == nullptr) {
    return;
  }
  if (data_chunk_buf_.Push(std::move(data)) && ++entry_nums_ % kNotifyInterval == 0) {
    std::lock_guard<std::mutex> lk(cv_buffer_empty_mtx_);
    cv_buffer_empty_.notify_one();
  }
}

void DataDumper::Dump(std::map<std::string, std::string> &dataMap) {
  std::ofstream file;
  for (auto &data : dataMap) {
    std::string dump_file = path_ + "/" + data.first;
    if (!Utils::IsFileExist(dump_file) && !Utils::CreateFile(dump_file)) {
      continue;
    }
    file.open(dump_file, std::ios::out | std::ios::app | std::ios::binary);
    if (!file.is_open()) {
      continue;
    }
    file.write(data.second.c_str(), data.second.size());
    file.close();
  }
}

void DataDumper::DataClassifyGather(std::map<std::string, std::string> &dataMap) {
  uint64_t batchSize = 0;
  while (batchSize < kBatchMaxLen) {
    std::unique_ptr<BaseReportData> data = nullptr;
    bool ret = data_chunk_buf_.Pop(data);
    if (!ret) {
      break;
    }
    if (data == nullptr) {
      return;
    }
    std::vector<uint8_t> encodeData = data->encode();
    std::string dataStr = std::string(reinterpret_cast<const char*>(encodeData.data()), encodeData.size());
    batchSize += dataStr.size();
    std::string key = data->tag;
    auto iter = dataMap.find(key);
    if (iter == dataMap.end()) {
      dataMap.insert({key, dataStr});
    } else {
      iter->second += dataStr;
    }
  }
}
} // profiler
} // toolkit
} // torch_npu
