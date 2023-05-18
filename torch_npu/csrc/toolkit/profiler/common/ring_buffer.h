#pragma once

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
    : is_inited_(false),
      is_quit_(false),
      read_index_(0),
      write_index_(0),
      idle_write_index_(0),
      use_extend_data_queue_(false),
      capacity_(0),
      mask_(0) {}

  ~RingBuffer() {
    if (is_inited_) {
      is_inited_ = false;
      is_quit_ = true;
      data_queue_.clear();
      extend_data_queue_.clear();
    }
  }

  void Init(size_t capacity) {
    capacity_ = capacity;
    mask_ = capacity_ - 1;
    data_queue_.resize(capacity);
    is_inited_ = true;
  }

  bool Push(T data) {
    if (use_extend_data_queue_.load(std::memory_order_relaxed) && is_inited_ && !is_quit_) {
      return PushExtendQueue(std::move(data));
    }
    size_t curr_read_index = 0;
    size_t curr_write_index = 0;
    size_t next_write_index = 0;
    size_t cycles = 0;
    do {
      if (!is_inited_ || is_quit_) {
        return false;
      }
      cycles++;
      if (cycles >= 1024) {
        return false;
      }
      curr_read_index = read_index_.load(std::memory_order_relaxed);
      curr_write_index = idle_write_index_.load(std::memory_order_relaxed);
      next_write_index = curr_write_index + 1;
      if ((next_write_index & mask_) == (curr_read_index & mask_)) {
        use_extend_data_queue_.store(true);
        return PushExtendQueue(std::move(data));
      }
    } while (!idle_write_index_.compare_exchange_weak(curr_write_index, next_write_index));
    size_t index = curr_write_index & mask_;
    data_queue_[index] = std::move(data);
    write_index_++;
    return true;
  }

  bool Pop(T &data) {
    if (!is_inited_) {
      return false;
    }
    size_t curr_read_index = read_index_.load(std::memory_order_relaxed);
    size_t curr_write_index = write_index_.load(std::memory_order_relaxed);
    if ((curr_read_index & mask_) == (curr_write_index & mask_)) {
      if (use_extend_data_queue_.load(std::memory_order_relaxed) && !is_quit_) {
        return PopExtendQueue(data);
      }
      return false;
    }
    size_t index = curr_read_index & mask_;
    data = std::move(data_queue_[index]);
    read_index_++;
    return true;
  }

  size_t Size() {
    size_t curr_read_index = read_index_.load(std::memory_order_relaxed);
    size_t curr_write_index = write_index_.load(std::memory_order_relaxed);
    size_t extend_queue_size = (use_extend_data_queue_.load(std::memory_order_relaxed) == true) ? ExtendSize() : 0;
    if (curr_read_index > curr_write_index) {
      return capacity_ - (curr_read_index & mask_) + (curr_write_index & mask_) + extend_queue_size;
    }
    return curr_write_index - curr_read_index + extend_queue_size;
  }

private:
  bool PushExtendQueue(T data) {
    std::lock_guard<std::mutex> lk(extend_queue_mtx_);
    extend_data_queue_.push_back(std::move(data));
    return true;
  }

  bool PopExtendQueue(T &data) {
    std::lock_guard<std::mutex> lk(extend_queue_mtx_);
    if (extend_data_queue_.empty()) {
      return false;
    }
    data = std::move(extend_data_queue_.front());
    extend_data_queue_.pop_front();
    return true;
  }

  size_t ExtendSize() {
    std::lock_guard<std::mutex> lk(extend_queue_mtx_);
    return extend_data_queue_.size();
  }

private:
  bool is_inited_;
  volatile bool is_quit_;
  std::atomic<size_t> read_index_;
  std::atomic<size_t> write_index_;
  std::atomic<size_t> idle_write_index_;
  std::atomic<bool> use_extend_data_queue_;
  size_t capacity_;
  size_t mask_;
  std::vector<T> data_queue_;
  std::deque<T> extend_data_queue_;
  std::mutex extend_queue_mtx_;
};
} // profiler
} // toolkit
} // torch_npu
