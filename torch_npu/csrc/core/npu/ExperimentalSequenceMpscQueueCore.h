#pragma once

#include "torch_npu/csrc/core/npu/OrderedMpscQueueCore.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace c10_npu::queue {

// Bounded sequence MPSC ring. Producers copy into independently reserved slots;
// state transitions and the single consumer have explicit admission gates.
// Start/restart/destruction are exclusive lifecycle operations: callers must
// stop and join old producer/consumer actors before starting a new generation
// or destroying the object. Administrative methods must not run in a callback.
// A generation fails before ticket/tag arithmetic can wrap; rollover is not a
// supported continuation of the generation and requires the same joined restart.
class ExperimentalSequenceMpscQueueCore final {
 public:
  explicit ExperimentalSequenceMpscQueueCore(size_t capacity)
      : capacity_(CheckedCapacity(capacity)), slots_(new Slot[capacity_]) {
    ResetSequences();
  }
  ~ExperimentalSequenceMpscQueueCore() { StopAndDiscardMetadata(); }
  ExperimentalSequenceMpscQueueCore(const ExperimentalSequenceMpscQueueCore&) = delete;
  ExperimentalSequenceMpscQueueCore& operator=(const ExperimentalSequenceMpscQueueCore&) = delete;

  size_t Capacity() const noexcept { return capacity_; }

  void Start() {
    std::lock_guard<std::mutex> state(state_mu_);
    std::unique_lock<std::mutex> consumer(consumer_mu_, std::try_to_lock);
    if (mode_.load() != QueueMode::Stopped || active_producers_.load(std::memory_order_seq_cst) != 0 || !consumer.owns_lock()) {
      throw std::logic_error("sequence queue can only start after its old actors have stopped");
    }
    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
    completed_pos_.store(0, std::memory_order_relaxed);
    ResetSequences();
    error_ = nullptr;
    failed_.store(false, std::memory_order_release);
    mode_.store(QueueMode::Running, std::memory_order_seq_cst);
  }

  template <typename Copy, typename WaitHooks>
  bool Enqueue(Copy&& copy, WaitHooks&& hooks) {
    ProducerLease producer(*this);
    if (!producer.admitted) { RethrowIfFailed(); return false; }
    size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    bool waiting = false;
    try {
      for (;;) {
        RethrowIfFailed();
        if (mode_.load(std::memory_order_seq_cst) != QueueMode::Running) {
          FinishWait(hooks, waiting);
          return false;
        }
        if (pos > std::numeric_limits<size_t>::max() - capacity_) {
          const auto error = std::make_exception_ptr(std::overflow_error("sequence queue ticket generation exhausted"));
          set_failed(error);
          std::rethrow_exception(error);
        }
        Slot& slot = slots_[pos % capacity_];
        const size_t sequence = slot.sequence.load(std::memory_order_acquire);
        const size_t difference = sequence - pos;
        if (difference == 0) {
          // Restore the runtime before reserving a ticket. A throwing hook must
          // never leave an unpublished hole at the head of the sequence ring.
          if (waiting) {
            FinishWait(hooks, waiting);
            pos = enqueue_pos_.load(std::memory_order_relaxed);
            continue;
          }
          if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
        } else if (difference > std::numeric_limits<size_t>::max() / 2) {
          BeginWait(hooks, waiting);
          std::this_thread::yield();
          pos = enqueue_pos_.load(std::memory_order_relaxed);
        } else {
          pos = enqueue_pos_.load(std::memory_order_relaxed);
        }
      }
      Slot& slot = slots_[pos % capacity_];
      slot.aborted.store(false, std::memory_order_relaxed);
      try {
        std::forward<Copy>(copy)(static_cast<uint32_t>(pos % capacity_));
      } catch (...) {
        const auto error = std::current_exception();
        set_failed(error);
        slot.aborted.store(true, std::memory_order_relaxed);
        slot.sequence.store(pos + 1, std::memory_order_release);
        std::rethrow_exception(error);
      }
      slot.sequence.store(pos + 1, std::memory_order_release);
      return true;
    } catch (...) {
      const auto error = std::current_exception();
      try { FinishWait(hooks, waiting); } catch (...) {}
      std::rethrow_exception(error);
    }
  }

  template <typename Consume>
  bool ConsumeOne(Consume&& consume) {
    NoopWaitHooks hooks;
    return ConsumeOne(std::forward<Consume>(consume), hooks);
  }

  template <typename Consume, typename WaitHooks>
  bool ConsumeOne(Consume&& consume, WaitHooks&& hooks) {
    // The one consumer can contend with a legitimate InspectHead/admin lease.
    // Waiting for that gate is not an additional data-consumer admission.
    auto consumer = WaitConsumer(hooks);
    return ConsumeOwned(std::forward<Consume>(consume), hooks);
  }

  template <typename WaitHooks>
  void WaitDrained(WaitHooks&& hooks) {
    bool waiting = false;
    try {
      for (;;) {
        RethrowIfFailed();
        // dequeue_pos_ denotes a claim, not completion of its callback.
        const size_t done = completed_pos_.load(std::memory_order_acquire);
        const size_t end = enqueue_pos_.load(std::memory_order_acquire);
        // A producer can publish and leave between the first position reads
        // and the SC zero count. Re-read its publication after that acquire.
        if (done == end && active_producers_.load(std::memory_order_seq_cst) == 0 &&
            enqueue_pos_.load(std::memory_order_acquire) == end) break;
        BeginWait(hooks, waiting);
        std::this_thread::yield();
      }
      FinishWait(hooks, waiting);
      RethrowIfFailed();
    } catch (...) {
      const auto error = std::current_exception();
      try { FinishWait(hooks, waiting); } catch (...) {}
      std::rethrow_exception(error);
    }
  }

  void RequestStop() {
    std::lock_guard<std::mutex> state(state_mu_);
    if (mode_.load(std::memory_order_relaxed) == QueueMode::Running) {
      mode_.store(QueueMode::Stopping, std::memory_order_seq_cst);
    }
  }

  void StopAndDrain() { NoopWaitHooks hooks; StopAndDrain(hooks); }
  template <typename WaitHooks>
  void StopAndDrain(WaitHooks&& hooks) {
    if (mode_.load(std::memory_order_acquire) == QueueMode::Stopped) return;
    RequestStop();
    WaitProducers(hooks);
    WaitDrained(hooks);
    auto consumer = WaitConsumer(hooks);
    RethrowIfFailed();
    MarkStopped();
  }

  void StopAndDiscard() { StopAndDiscardMetadata(); }
  void StopAndDiscardMetadata() { NoopWaitHooks hooks; StopAndDiscardMetadata(hooks); }
  template <typename WaitHooks>
  void StopAndDiscardMetadata(WaitHooks&& hooks) {
    RequestStop();
    WaitProducers(hooks);
    auto consumer = WaitConsumer(hooks);
    const size_t end = enqueue_pos_.load(std::memory_order_acquire);
    dequeue_pos_.store(end, std::memory_order_release);
    completed_pos_.store(end, std::memory_order_release);
    MarkStopped();
  }

  template <typename Discard>
  void StopAndDiscard(Discard&& discard) {
    NoopWaitHooks hooks;
    StopAndDiscard(std::forward<Discard>(discard), hooks);
  }
  template <typename Discard, typename WaitHooks>
  void StopAndDiscard(Discard&& discard, WaitHooks&& hooks) {
    RequestStop();
    WaitProducers(hooks);
    auto consumer = WaitConsumer(hooks);
    RethrowIfFailed();
    while (completed_pos_.load(std::memory_order_acquire) != enqueue_pos_.load(std::memory_order_acquire)) {
      ConsumeOwned(discard, hooks);
    }
    MarkStopped();
  }

  void Fail(std::exception_ptr error) { set_failed(error); }
  void RethrowIfFailed() const {
    if (!failed_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> state(state_mu_);
    if (error_) std::rethrow_exception(error_);
  }
  QueueSnapshot Snapshot() const {
    const size_t enq = enqueue_pos_.load(std::memory_order_acquire);
    const size_t done = completed_pos_.load(std::memory_order_acquire);
    const size_t size = enq >= done ? std::min(enq - done, capacity_) : 0;
    return QueueSnapshot{capacity_, size, active_producers_.load(), consumer_active_.load(),
                         mode_.load(), failed_.load()};
  }
  template <typename Inspect>
  bool InspectHead(Inspect&& inspect) const {
    std::unique_lock<std::mutex> consumer(consumer_mu_, std::try_to_lock);
    if (!consumer.owns_lock()) return false;
    const size_t pos = dequeue_pos_.load(std::memory_order_acquire);
    if (pos == enqueue_pos_.load(std::memory_order_acquire) ||
        slots_[pos % capacity_].sequence.load(std::memory_order_acquire) != pos + 1) return false;
    std::forward<Inspect>(inspect)(static_cast<uint32_t>(pos % capacity_));
    return true;
  }

 private:
  class ProducerLease {
   public:
    explicit ProducerLease(ExperimentalSequenceMpscQueueCore& core) : core_(core) {
      // Registration and closing share a single SC order: an actor that saw
      // Running registered before the stop store, so stop cannot miss it.
      // A later registrant sees the closed mode and never touches a slot.
      core_.active_producers_.fetch_add(1, std::memory_order_seq_cst);
      admitted = core_.mode_.load(std::memory_order_seq_cst) == QueueMode::Running;
    }
    ~ProducerLease() {
      core_.active_producers_.fetch_sub(1, std::memory_order_seq_cst);
    }
    bool admitted = false;
   private:
    ExperimentalSequenceMpscQueueCore& core_;
  };

  static size_t CheckedCapacity(size_t capacity) {
    if (capacity < 2 || capacity > static_cast<size_t>(UINT32_MAX) ||
        capacity > std::numeric_limits<size_t>::max() / 2) {
      throw std::invalid_argument("sequence MPSC queue capacity must be between2 and UINT32_MAX");
    }
    return capacity;
  }
  void ResetSequences() {
    for (size_t i = 0; i < capacity_; ++i) {
      slots_[i].sequence.store(i, std::memory_order_relaxed);
      slots_[i].aborted.store(false, std::memory_order_relaxed);
    }
  }
  template <typename WaitHooks>
  static void BeginWait(WaitHooks& hooks, bool& waiting) {
    if (!waiting) { hooks.BeforeWait(); waiting = true; }
  }
  template <typename WaitHooks>
  static void FinishWait(WaitHooks& hooks, bool& waiting) {
    if (waiting) { waiting = false; hooks.AfterWait(); }
  }
  template <typename WaitHooks>
  void WaitProducers(WaitHooks& hooks) {
    bool waiting = false;
    try {
      while (active_producers_.load(std::memory_order_seq_cst) != 0) {
        BeginWait(hooks, waiting);
        std::this_thread::yield();
      }
      FinishWait(hooks, waiting);
    } catch (...) {
      const auto error = std::current_exception();
      try { FinishWait(hooks, waiting); } catch (...) {}
      std::rethrow_exception(error);
    }
  }
  template <typename WaitHooks>
  std::unique_lock<std::mutex> WaitConsumer(WaitHooks& hooks) {
    std::unique_lock<std::mutex> consumer(consumer_mu_, std::defer_lock);
    bool waiting = false;
    try {
      for (;;) {
        if (consumer.try_lock()) {
          if (!waiting) return consumer;
          consumer.unlock(); // Restore a language runtime outside queue locks.
          FinishWait(hooks, waiting);
        } else {
          BeginWait(hooks, waiting);
          std::this_thread::yield();
        }
      }
    } catch (...) {
      const auto error = std::current_exception();
      try { FinishWait(hooks, waiting); } catch (...) {}
      std::rethrow_exception(error);
    }
  }
  void MarkStopped() {
    std::lock_guard<std::mutex> state(state_mu_);
    mode_.store(QueueMode::Stopped, std::memory_order_seq_cst);
  }
  void Retire(size_t pos) {
    slots_[pos % capacity_].sequence.store(pos + capacity_, std::memory_order_release);
    completed_pos_.store(pos + 1, std::memory_order_release);
    consumer_active_.store(false, std::memory_order_release);
  }
  template <typename Consume, typename WaitHooks>
  bool ConsumeOwned(Consume&& consume, WaitHooks& hooks) {
    size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    bool waiting = false, claimed = false;
    try {
      for (;;) {
        RethrowIfFailed();
        Slot& slot = slots_[pos % capacity_];
        const size_t sequence = slot.sequence.load(std::memory_order_acquire);
        const size_t difference = sequence - (pos + 1);
        if (difference == 0) {
          consumer_active_.store(true, std::memory_order_release);
          dequeue_pos_.store(pos + 1, std::memory_order_relaxed);
          claimed = true;
          break;
        }
        if (difference > std::numeric_limits<size_t>::max() / 2) {
          if (mode_.load(std::memory_order_seq_cst) != QueueMode::Running &&
              active_producers_.load(std::memory_order_seq_cst) == 0) {
            // The last producer may have published after our first sequence
            // read and before leaving. Do not exit past that committed item.
            if (slot.sequence.load(std::memory_order_acquire) == pos + 1) continue;
            FinishWait(hooks, waiting);
            RethrowIfFailed();
            return false;
          }
          BeginWait(hooks, waiting);
          std::this_thread::yield();
        }
        pos = dequeue_pos_.load(std::memory_order_relaxed);
      }
      FinishWait(hooks, waiting);
      if (slots_[pos % capacity_].aborted.load(std::memory_order_acquire)) {
        RethrowIfFailed();
        throw std::runtime_error("sequence queue aborted slot");
      }
      std::forward<Consume>(consume)(static_cast<uint32_t>(pos % capacity_));
      Retire(pos);
      return true;
    } catch (...) {
      const auto error = std::current_exception();
      try { FinishWait(hooks, waiting); } catch (...) {}
      set_failed(error);
      if (claimed) Retire(pos);
      std::rethrow_exception(error);
    }
  }
  void set_failed(std::exception_ptr error) {
    std::lock_guard<std::mutex> state(state_mu_);
    if (!error_) error_ = error ? error : std::make_exception_ptr(std::runtime_error("sequence queue failed"));
    failed_.store(true, std::memory_order_release);
    mode_.store(QueueMode::Failed, std::memory_order_seq_cst);
  }
  struct alignas(64) Slot { std::atomic<size_t> sequence{0}; std::atomic<bool> aborted{false}; };
  const size_t capacity_;
  std::unique_ptr<Slot[]> slots_;
  std::atomic<size_t> enqueue_pos_{0}, dequeue_pos_{0}, completed_pos_{0}, active_producers_{0};
  std::atomic<bool> consumer_active_{false}, failed_{false};
  std::atomic<QueueMode> mode_{QueueMode::Stopped};
  mutable std::mutex state_mu_, consumer_mu_;
  std::exception_ptr error_;
};

} // namespace c10_npu::queue
