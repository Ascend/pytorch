#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <condition_variable>

#include "torch_npu/csrc/toolkit/profiler/common/thread.h"
#include "torch_npu/csrc/toolkit/profiler/common/ring_buffer.h"
#include "torch_npu/csrc/toolkit/profiler/inc/data_reporter.h"

namespace torch_npu {
namespace toolkit {
namespace profiler {
constexpr uint32_t kDefaultRingBuffer = 1024;
constexpr uint32_t kBatchMaxLen = 5 * 1024 * 1024; // 5 MB
constexpr uint32_t kMaxWaitTimeUs = 1024;
constexpr uint32_t kNotifyInterval = 256;

class DataDumper : public Thread {
public:
  explicit DataDumper();
  virtual ~DataDumper();
  void Init(const std::string &path, size_t capacity);
  void UnInit();
  void Report(std::unique_ptr<BaseReportData> data);
  void Start();
  void Stop();

private:
  void Flush();
  void Dump(const std::map<std::string, std::vector<uint8_t>> &dataMap);
  void Run();
  void GatherAndDumpData();

private:
  std::string path_;
  std::atomic<bool> start_;
  std::atomic<bool> init_;
  RingBuffer<std::unique_ptr<BaseReportData>> data_chunk_buf_;
  std::map<std::string, FILE*> fd_map_;
};
} // profiler
} // toolkit
} // torch_npu
