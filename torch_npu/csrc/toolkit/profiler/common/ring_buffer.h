// Copyright (c) 2023, Huawei Technologies.All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef TORCH_NPU_TOOLKIT_PROFILER_RING_BUFFER_INC
#define TORCH_NPU_TOOLKIT_PROFILER_RING_BUFFER_INC

#include <atomic>
#include <vector>
#include <deque>

namespace torch_npu {
namespace toolkit {
namespace profiler {
template <typename T>
class RingBuffer {
public:
  RingBuffer()
    : isInited_(false),
      isQuit_(false),
      readIndex_(0),
      writeIndex_(0),
      idleWriteIndex_(0),
      useExtendDataQueue_(false),
      capacity_(0),
      mask_(0) {}

  ~RingBuffer() {
    if (isInited_) {
      isInited_ = true;
      isQuit_ = true;
      dataQueue_.clear();
      extendDataQueue_.clear();
    }
  }

  void Init(size_t capacity) {
    capacity_ = capacity;
    mask_ = capacity_ - 1;
    dataQueue_.resize(capacity);
    isInited_ = true;
  }

  bool Push(T data) {
    if (useExtendDataQueue_.load(std::memory_order_relaxed) && isInited_ && !isQuit_) {
      return PushExtendQueue(std::move(data));
    }
    size_t currReadIndex = 0;
    size_t currWriteIndex = 0;
    size_t nextWriteIndex = 0;
    size_t cycles = 0;
    do {
      if (!isInited_ || isQuit_) {
        return false;
      }
      cycles++;
      if (cycles >= 1024) {
        return false;
      }
      currReadIndex = readIndex_.load(std::memory_order_relaxed);
      currWriteIndex = idleWriteIndex_.load(std::memory_order_relaxed);
      nextWriteIndex = currWriteIndex + 1;
      if ((nextWriteIndex & mask_) == (currReadIndex & mask_)) {
        useExtendDataQueue_.store(true);
        return PushExtendQueue(std::move(data));
      }
    } while (!idleWriteIndex_.compare_exchange_weak(currWriteIndex, nextWriteIndex));
    size_t index = currWriteIndex & mask_;
    dataQueue_[index] = std::move(data);
    writeIndex_++;
    return true;
  }

  bool Pop(T &data) {
    if (!isInited_) {
      return false;
    }
    size_t currReadIndex = readIndex_.load(std::memory_order_relaxed);
    size_t currWriteIndex = writeIndex_.load(std::memory_order_relaxed);
    if ((currReadIndex & mask_) == (currWriteIndex & mask_)) {
      if (useExtendDataQueue_.load(std::memory_order_relaxed) && !isQuit_) {
        return PopExtendQueue(data);
      }
      return false;
    }
    size_t index = currReadIndex & mask_;
    data = std::move(dataQueue_[index]);
    readIndex_++;
    return true;
  }

  size_t Size() {
    size_t currReadIndex = readIndex_.load(std::memory_order_relaxed);
    size_t currWriteIndex = writeIndex_.load(std::memory_order_relaxed);
    size_t extendQueueSize = (useExtendDataQueue_.load(std::memory_order_relaxed) == true) ? ExtendSize() : 0;
    if (currReadIndex > currWriteIndex) {
      return capacity_ - (currReadIndex & mask_) + (currWriteIndex & mask_) + extendQueueSize;
    }
    return currWriteIndex - currReadIndex + extendQueueSize;
  }

private:
  bool PushExtendQueue(T data) {
    std::lock_guard<std::mutex> lk(extendQueueMtx_);
    extendDataQueue_.push_back(std::move(data));
    return true;
  }

  bool PopExtendQueue(T &data) {
    std::lock_guard<std::mutex> lk(extendQueueMtx_);
    if (extendDataQueue_.empty()) {
      return false;
    }
    data = std::move(extendDataQueue_.front());
    extendDataQueue_.pop_front();
    return true;
  }

  size_t ExtendSize() {
    std::lock_guard<std::mutex> lk(extendQueueMtx_);
    return extendDataQueue_.size();
  }

private:
  bool isInited_;
  volatile bool isQuit_;
  std::atomic<size_t> readIndex_;
  std::atomic<size_t> writeIndex_;
  std::atomic<size_t> idleWriteIndex_;
  std::atomic<bool> useExtendDataQueue_;
  size_t capacity_;
  size_t mask_;
  std::vector<T> dataQueue_;
  std::deque<T> extendDataQueue_;
  std::mutex extendQueueMtx_;
};
} // profiler
} // toolkit
} // torch_npu

#endif
