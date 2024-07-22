#include <unistd.h>

#include <vector>
#include <map>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

#include "torch_npu/csrc/toolkit/profiler/inc/data_dumper.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/core/npu/npu_log.h"

namespace torch_npu {
namespace toolkit {
namespace profiler {
DataDumper::DataDumper()
    : path_(""),
      start_(false),
      init_(false) {}

DataDumper::~DataDumper() {
  UnInit();
}

void DataDumper::Init(const std::string &path, size_t capacity = kDefaultRingBuffer) {
  path_ = path;
  data_chunk_buf_.Init(capacity);
  init_.store(true);
}

void DataDumper::UnInit() {
    if (init_.load()) {
        data_chunk_buf_.UnInit();
        init_.store(false);
        start_.store(false);
        for (auto &f : fd_map_) {
            if (f.second != nullptr) {
                fclose(f.second);
                f.second = nullptr;
            }
        }
        fd_map_.clear();
    }
}

void DataDumper::Start() {
    if (!init_.load() || Thread::Start() != 0) {
        return;
    }
    start_.store(true);
}

void DataDumper::Stop() {
  if (start_.load() == true) {
    start_.store(false);
    Thread::Stop();
  }
  Flush();
}

void DataDumper::GatherAndDumpData() {
    std::map<std::string, std::vector<uint8_t>> dataMap;
    uint64_t batchSize = 0;
    while (batchSize < kBatchMaxLen) {
        std::unique_ptr<BaseReportData> data = nullptr;
        if (!data_chunk_buf_.Pop(data) || data == nullptr) {
            break;
        }
        std::vector<uint8_t> encodeData = data->encode();
        batchSize += encodeData.size();
        const std::string &key = data->tag;
        auto iter = dataMap.find(key);
        if (iter == dataMap.end()) {
            dataMap.insert({key, encodeData});
        } else {
            iter->second.insert(iter->second.end(), encodeData.cbegin(), encodeData.cend());
        }
    }
    if (dataMap.size() > 0) {
        static bool create_flag = true;
        if (create_flag) {
            create_flag = !Utils::CreateDir(this->path_);
        }
        Dump(dataMap);
    }
}

void DataDumper::Run() {
    for (;;) {
        if (!start_.load()) {
            break;
        }
        if (data_chunk_buf_.Size() > kNotifyInterval) {
            GatherAndDumpData();
        } else {
            usleep(kMaxWaitTimeUs);
        }
    }
}

void DataDumper::Flush() {
  while (data_chunk_buf_.Size() != 0) {
    GatherAndDumpData();
  }
}

void DataDumper::Report(std::unique_ptr<BaseReportData> data)
{
    if (C10_UNLIKELY(!start_.load() || data == nullptr)) {
        return;
    }
    if (!data_chunk_buf_.Push(std::move(data))) {
        ASCEND_LOGE("DataDumper report data failed");
    }
}

void DataDumper::Dump(const std::map<std::string, std::vector<uint8_t>> &dataMap)
{
    for (auto &data : dataMap) {
        FILE *fd = nullptr;
        const std::string dump_file = path_ + "/" + data.first;
        auto iter = fd_map_.find(dump_file);
        if (iter == fd_map_.end()) {
            if (!Utils::IsFileExist(dump_file) && !Utils::CreateFile(dump_file)) {
                continue;
            }
            fd = fopen(dump_file.c_str(), "ab");
            if (fd == nullptr) {
                ASCEND_LOGE("DataDumper open file failed: %s", dump_file.c_str());
                continue;
            }
            fd_map_.insert({dump_file, fd});
        } else {
            fd = iter->second;
        }
        fwrite(reinterpret_cast<const char*>(data.second.data()), sizeof(char), data.second.size(), fd);
    }
}
} // profiler
} // toolkit
} // torch_npu
