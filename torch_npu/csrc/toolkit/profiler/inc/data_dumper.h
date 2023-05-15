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

class DataDumper : public Thread {
public:
  explicit DataDumper();
  virtual ~DataDumper();
  void Init(const std::string &path, size_t capacity);
  void Flush();
  void Report(std::unique_ptr<BaseReportData> data);
  void Start();
  void Stop();

private:
  void Dump(std::map<std::string, std::string> &dataMap);
  void Run();
  void DataClassifyGather(std::map<std::string, std::string> &dataMap);
  void SetBufferEmptyEvent();
  void WaitBufferEmptyEvent(uint64_t us);

private:
  std::string path_;
  std::atomic<bool> start_;
  std::atomic<uint32_t> entry_nums_;
  std::mutex cv_buffer_empty_mtx_;
  std::condition_variable cv_buffer_empty_;
  RingBuffer<std::unique_ptr<BaseReportData>> data_chunk_buf_;
};
} // profiler
} // toolkit
} // torch_npu
