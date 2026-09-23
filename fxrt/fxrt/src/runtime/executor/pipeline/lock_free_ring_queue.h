#ifndef __RUNTIME_EXECUTOR_PIPELINE_LOCK_FREE_RING_QUEUE_H__
#define __RUNTIME_EXECUTOR_PIPELINE_LOCK_FREE_RING_QUEUE_H__

#include <atomic>
#include <array>
#include <utility>
#include <cstddef>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace fxrt {
namespace runtime {
constexpr size_t kQueueElemAlignSize = 128;
// This is a lock-free queue implementation that supports multiple producers and a single consumer, improves performance
// through spinning, and supports inplacement construct element, pause and continue the queue, the queue is in pause
// status after creation.
template <typename T, uint64_t Capacity>
class LockFreeRingQueue {
  static_assert(Capacity > 0, "Capacity must be greater than 0 for LockFreeRingQueue");
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
  static_assert(std::is_nothrow_move_constructible_v<T>, "The template type T must has nothrow move constructible");
  static_assert(std::is_nothrow_destructible_v<T>, "The template type T must has nothrow destructible");

 public:
  LockFreeRingQueue() = default;
  ~LockFreeRingQueue() = default;

  LockFreeRingQueue(const LockFreeRingQueue&) = delete;
  LockFreeRingQueue& operator=(const LockFreeRingQueue&) = delete;

  // The Element is the element type stored in buffer_.
  struct Element {
    std::atomic<bool> ready{false};
    typename std::aligned_storage<sizeof(T), alignof(T)>::type storage;
  };

  // Pause the queue, can not push element to a queue which is in pause status, the queue is in pause status after
  // creation.
  void Pause() {
    running_.store(false, std::memory_order_release);
  }

  // Check the queue is on pause state.
  bool IsPaused() const {
    return false == running_.load(std::memory_order_acquire);
  }

  // Continue the queue which is in pause status.
  void Continue() {
    running_.store(true);

    std::unique_lock<std::mutex> lock(mtx_);
    cvPause_.notify_one();
  }

  void Finalize() noexcept {
    alive_.store(false);
    std::lock_guard<std::mutex> lock(mtx_);
    cvPause_.notify_all();
  }

  // Check the queue is empty or not.
  bool Empty() const noexcept {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
  }

  // Push element to lock free queue, the args parameter must be of type T or can construct type T object.
  // type. Push is multi thread safety and return until finishing push.
  template <typename... Args>
  bool Push(Args&&... args) {
    uint64_t currentTail;
    Element* currentElement;

    // Spin until the element is pushed, the queue is closed or it gets paused.
    while (alive_.load(std::memory_order_acquire)) {
      if (!running_.load(std::memory_order_acquire)) {
        RT_GLOG(ERROR) << "The queue is in pause status, can not push task.";
        return false;
      }

      if (TryPush(currentTail, currentElement, std::forward<Args>(args)...)) {
        return true;
      }
    }
    return false;
  }

  // Pop a element from buffer_, return until finishing pop.
  bool Pop() {
    // Spin until an element is popped or the queue is closed.
    while (alive_.load(std::memory_order_acquire)) {
      if (TryPop()) {
        return true;
      }
    }
    return false;
  }

  // Get the first element in queue, return until finishing get the first element.
  T* Front() noexcept {
    // Spin until an element shows up or the queue is closed.
    while (alive_.load(std::memory_order_acquire)) {
      if (T* ptr = TryFront()) {
        return ptr;
      }
      if (!running_.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lock(mtx_);
        cvPause_.wait(lock, [this] {
          return !Empty() || !alive_.load(std::memory_order_acquire) || running_.load(std::memory_order_acquire);
        });
      }
    }
    return nullptr;
  }

 private:
  template <typename... Args>
  bool TryPush(
      uint64_t& currentTail,
      Element*& element, // NOLINT(runtime/references)
      Args&&... args) {
    currentTail = tail_.load(std::memory_order_relaxed);
    auto currentHead = head_.load(std::memory_order_relaxed);
    if (currentTail < currentHead || currentTail - currentHead >= Capacity) {
      return false;
    }
    if (currentTail == UINT64_MAX) {
      // This is a safety check, it requires continuous pushing for millions of years to trigger an overflow.
      RT_GLOG(EXCEPTION) << "The queue is overflow and push task failed.";
    }

    if (!tail_.compare_exchange_weak(
            currentTail, currentTail + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      return false;
    }

    element = &buffer_[currentTail & mask_];
    new (&element->storage) T(std::forward<Args>(args)...);
    element->ready.store(true, std::memory_order_release);
    return true;
  }

  bool TryPop() {
    const uint64_t currentHead = head_.load(std::memory_order_relaxed);
    if (currentHead == tail_.load(std::memory_order_acquire)) {
      return false;
    }

    Element& element = buffer_[currentHead & mask_];
    if (!element.ready.load(std::memory_order_acquire)) {
      return false;
    }

    reinterpret_cast<T*>(&element.storage)->~T();
    element.ready.store(false, std::memory_order_release);
    head_.store(currentHead + 1, std::memory_order_release);
    return true;
  }

  T* TryFront() noexcept {
    const uint64_t currentHead = head_.load(std::memory_order_relaxed);
    if (currentHead == tail_.load(std::memory_order_acquire)) {
      return nullptr;
    }

    Element& element = buffer_[currentHead & mask_];
    if (!element.ready.load(std::memory_order_acquire)) {
      return nullptr;
    }

    return reinterpret_cast<T*>(&element.storage);
  }

  // Bitmask for fast modulo.
  static constexpr uint64_t mask_ = Capacity - 1;
  alignas(kQueueElemAlignSize) Element buffer_[Capacity];
  alignas(kQueueElemAlignSize) std::atomic<uint64_t> head_{0};
  alignas(kQueueElemAlignSize) std::atomic<uint64_t> tail_{0};

  // Indicates whether the queue is currently paused or in execution state.
  alignas(kQueueElemAlignSize) std::atomic<bool> running_{false};
  // Indicates whether the queue is currently in a closed state.
  alignas(kQueueElemAlignSize) std::atomic<bool> alive_{true};

  // Lock the status for pause or execution state.
  std::mutex mtx_;
  std::condition_variable cvPause_;
};
} // namespace runtime
} // namespace fxrt
#endif // __RUNTIME_EXECUTOR_PIPELINE_LOCK_FREE_RING_QUEUE_H__
