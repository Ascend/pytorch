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
        capacity_(0),
        mask_(0) {}

  ~RingBuffer() {
    UnInit();
  }

  void Init(size_t capacity) {
    capacity_ = capacity;
    mask_ = capacity_ - 1;
    data_queue_.resize(capacity);
    is_inited_ = true;
    is_quit_ = false;
  }

  void UnInit() {
    if (is_inited_) {
      data_queue_.clear();
      read_index_ = 0;
      write_index_ = 0;
      idle_write_index_ = 0;
      capacity_ = 0;
      mask_ = 0;
      is_quit_ = true;
      is_inited_ = false;
    }
  }

  bool Push(T data) {
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
        return false;
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
    if ((curr_read_index & mask_) == (curr_write_index & mask_) && !is_quit_) {
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
    if (curr_read_index > curr_write_index) {
      return capacity_ - (curr_read_index & mask_) + (curr_write_index & mask_);
    }
    return curr_write_index - curr_read_index;
  }

private:
  bool is_inited_;
  volatile bool is_quit_;
  std::atomic<size_t> read_index_;
  std::atomic<size_t> write_index_;
  std::atomic<size_t> idle_write_index_;
  size_t capacity_;
  size_t mask_;
  std::vector<T> data_queue_;
};
} // profiler
} // toolkit
} // torch_npu
