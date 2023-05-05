#include <unistd.h>

#include <vector>
#include <map>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "torch_npu/csrc/toolkit/profiler/inc/data_dumper.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"

namespace torch_npu {
namespace toolkit {
namespace profiler {
DataDumper::DataDumper()
  : path_(""),
    start_(false) {}

DataDumper::~DataDumper() {
  start_.store(false);
}

void DataDumper::Init(const std::string &path, size_t capacity = DEFAULT_RING_BUFFER) {
  path_ = path;
  dataChunkBuf_.Init(capacity);
}

void DataDumper::start() {
  bool ret = Utils::CreateDir(path_);
  if (ret != true) {
    return;
  }
  if (Thread::Start() != 0) {
    return;
  }
  start_.store(true);
}

void DataDumper::stop() {
  if (start_.load() == true) {
    start_.store(false);
    Thread::Stop();
  }
}

void DataDumper::setBufferEmptyEvent() {
  std::lock_guard<std::mutex> lk(cv_buffer_empty_mtx_);
  cv_buffer_empty_.notify_all();
}

void DataDumper::waitBufferEmptyEvent(uint64_t us) {
  std::unique_lock<std::mutex> lk(cv_buffer_empty_mtx_);
  cv_buffer_empty_.wait_for(lk, std::chrono::microseconds(us));
}

void DataDumper::Run() {
  std::map<std::string, std::string> dataMap;
  for (;;) {
    dataMap.clear();
    dataClassifyGather(dataMap);
    size_t size = dataMap.size();
    if (size == 0) {
      setBufferEmptyEvent();
      if (start_.load() != true) {
        break;
      }
      if (entry_nums_ >= 5) {
        entry_nums_ = 0;
      } else {
        usleep(1000);
      }
      continue;
    }
    dump(dataMap);
  }
}

void DataDumper::flush() {
  if (!start_.load()) {
    return;
  }
  waitBufferEmptyEvent(5000000);
}

void DataDumper::report(std::unique_ptr<BaseReportData> data) {
  if (!start_.load() || data == nullptr) {
    return;
  }
  if (dataChunkBuf_.Push(std::move(data))) {
    entry_nums_++;
  }
}

void DataDumper::dump(std::map<std::string, std::string> &dataMap) {
  std::ofstream file;
  for (auto &data : dataMap) {
    std::string dump_file = path_ + "/" + data.first;
    if (!Utils::IsFileExist(dump_file)) {
      int new_file = creat(dump_file, S_IRUSR|S_IWUSR|S_IROTH);
      close(new_file);
    }
    file.open(dump_file, std::ios::out | std::ios::app | std::ios::binary);
    if (!file.is_open()) {
      continue;
    }
    file.write(data.second.c_str(), data.second.size());
    file.close();
  }
}

void DataDumper::dataClassifyGather(std::map<std::string, std::string> &dataMap) {
  uint64_t batchSize = 0;
  while (batchSize < BATCH_MAX_LEN) {
    std::unique_ptr<BaseReportData> data = nullptr;
    bool ret = dataChunkBuf_.Pop(data);
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
