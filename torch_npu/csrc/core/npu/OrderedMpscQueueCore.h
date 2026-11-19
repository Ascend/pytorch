#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace c10_npu::queue {

enum class QueueMode {
    Stopped,
    Running,
    Stopping,
    Failed,
};

struct NoopWaitHooks {
    void BeforeWait() noexcept {}
    void AfterWait() noexcept {}
};

struct QueueSnapshot {
    size_t capacity = 0;
    size_t size = 0;
    size_t active_producers = 0;
    bool consumer_active = false;
    QueueMode mode = QueueMode::Stopped;
    bool failed = false;
};

class OrderedMpscQueueCore final {
public:
    explicit OrderedMpscQueueCore(size_t capacity) : capacity_(capacity)
    {
        if (capacity_ == 0 || capacity_ > static_cast<size_t>(UINT32_MAX)) {
            throw std::invalid_argument("ordered MPSC queue capacity is out of range");
        }
    }

    ~OrderedMpscQueueCore()
    {
        StopAndDiscard();
    }

    OrderedMpscQueueCore(const OrderedMpscQueueCore&) = delete;
    OrderedMpscQueueCore& operator=(const OrderedMpscQueueCore&) = delete;

    size_t Capacity() const noexcept
    {
        return capacity_;
    }

    void Start()
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (mode_ != QueueMode::Stopped || active_producers_ != 0 || consumer_active_) {
            throw std::logic_error("ordered MPSC queue can only start from a stopped state");
        }
        head_ = 0;
        tail_ = 0;
        size_ = 0;
        error_ = nullptr;
        mode_ = QueueMode::Running;
        NotifyAllLocked();
    }

    template <typename Copy, typename WaitHooks>
    bool Enqueue(Copy&& copy, WaitHooks&& wait_hooks)
    {
        std::unique_lock<std::mutex> lock(mu_);
        RethrowIfFailedLocked();
        if (mode_ != QueueMode::Running) {
            return false;
        }

        ++active_producers_;
        auto leave_producer = [this]() {
            --active_producers_;
            if (active_producers_ == 0) {
                cv_no_active_producers_.notify_all();
            }
            NotifyDrainedIfLocked();
        };

        try {
            while (size_ == capacity_ && mode_ == QueueMode::Running && error_ == nullptr) {
                WaitOnce(cv_not_full_, lock, wait_hooks);
            }
            RethrowIfFailedLocked();
            if (mode_ != QueueMode::Running) {
                leave_producer();
                return false;
            }

            const uint32_t slot = tail_;
            std::forward<Copy>(copy)(slot);
            tail_ = Next(tail_);
            ++size_;
            leave_producer();
            cv_not_empty_.notify_one();
            return true;
        } catch (...) {
            leave_producer();
            throw;
        }
    }

    template <typename Consume>
    bool ConsumeOne(Consume&& consume)
    {
        NoopWaitHooks wait_hooks;
        return ConsumeOne(std::forward<Consume>(consume), wait_hooks);
    }

    template <typename Consume, typename WaitHooks>
    bool ConsumeOne(Consume&& consume, WaitHooks&& wait_hooks)
    {
        uint32_t slot = 0;
        {
            std::unique_lock<std::mutex> lock(mu_);
            auto ready = [this] {
                return error_ != nullptr ||
                    (!consumer_active_ && size_ != 0) ||
                    (size_ == 0 && mode_ != QueueMode::Running);
            };
            while (!ready()) {
                WaitOnce(cv_not_empty_, lock, wait_hooks);
            }
            RethrowIfFailedLocked();
            if (size_ == 0) {
                return false;
            }
            consumer_active_ = true;
            slot = head_;
        }

        try {
            std::forward<Consume>(consume)(slot);
        } catch (...) {
            const std::exception_ptr callback_error = std::current_exception();
            {
                std::lock_guard<std::mutex> lock(mu_);
                consumer_active_ = false;
                if (error_ == nullptr) {
                    error_ = callback_error;
                }
                mode_ = QueueMode::Failed;
                NotifyAllLocked();
            }
            std::rethrow_exception(callback_error);
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            head_ = Next(head_);
            --size_;
            consumer_active_ = false;
            cv_not_full_.notify_all();
            cv_not_empty_.notify_all();
            cv_drained_.notify_all();
        }
        return true;
    }

    template <typename WaitHooks>
    void WaitDrained(WaitHooks&& wait_hooks)
    {
        std::unique_lock<std::mutex> lock(mu_);
        while (!IsDrainedLocked() && error_ == nullptr) {
            WaitOnce(cv_drained_, lock, wait_hooks);
        }
        RethrowIfFailedLocked();
    }

    void RequestStop()
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (mode_ == QueueMode::Running) {
            mode_ = QueueMode::Stopping;
        }
        NotifyAllLocked();
    }

    void StopAndDrain()
    {
        NoopWaitHooks wait_hooks;
        StopAndDrain(wait_hooks);
    }

    template <typename WaitHooks>
    void StopAndDrain(WaitHooks&& wait_hooks)
    {
        std::unique_lock<std::mutex> lock(mu_);
        if (mode_ == QueueMode::Stopped) {
            return;
        }
        if (mode_ == QueueMode::Running) {
            mode_ = QueueMode::Stopping;
        }
        NotifyAllLocked();

        while (active_producers_ != 0) {
            WaitOnce(cv_no_active_producers_, lock, wait_hooks);
        }
        RethrowIfFailedLocked();
        while (size_ != 0 || consumer_active_) {
            WaitOnce(cv_drained_, lock, wait_hooks);
            RethrowIfFailedLocked();
        }
        mode_ = QueueMode::Stopped;
        NotifyAllLocked();
    }

    void StopAndDiscard()
    {
        StopAndDiscardMetadata();
    }

    void StopAndDiscardMetadata()
    {
        NoopWaitHooks wait_hooks;
        StopAndDiscardMetadata(wait_hooks);
    }

    template <typename WaitHooks>
    void StopAndDiscardMetadata(WaitHooks&& wait_hooks)
    {
        std::unique_lock<std::mutex> lock(mu_);
        if (mode_ == QueueMode::Stopped) {
            return;
        }
        if (mode_ == QueueMode::Running) {
            mode_ = QueueMode::Stopping;
        }
        NotifyAllLocked();
        while (active_producers_ != 0) {
            WaitOnce(cv_no_active_producers_, lock, wait_hooks);
        }
        while (consumer_active_) {
            WaitOnce(cv_drained_, lock, wait_hooks);
        }
        head_ = 0;
        tail_ = 0;
        size_ = 0;
        mode_ = QueueMode::Stopped;
        NotifyAllLocked();
    }

    template <typename Discard>
    void StopAndDiscard(Discard&& discard)
    {
        NoopWaitHooks wait_hooks;
        StopAndDiscard(std::forward<Discard>(discard), wait_hooks);
    }

    template <typename Discard, typename WaitHooks>
    void StopAndDiscard(Discard&& discard, WaitHooks&& wait_hooks)
    {
        std::unique_lock<std::mutex> lock(mu_);
        RethrowIfFailedLocked();
        if (mode_ == QueueMode::Stopped) {
            return;
        }
        if (mode_ == QueueMode::Running) {
            mode_ = QueueMode::Stopping;
        }
        NotifyAllLocked();
        while (active_producers_ != 0) {
            WaitOnce(cv_no_active_producers_, lock, wait_hooks);
        }
        RethrowIfFailedLocked();
        while (consumer_active_) {
            WaitOnce(cv_drained_, lock, wait_hooks);
            RethrowIfFailedLocked();
        }

        while (size_ != 0) {
            const uint32_t slot = head_;
            consumer_active_ = true;
            lock.unlock();
            try {
                discard(slot);
            } catch (...) {
                const std::exception_ptr callback_error = std::current_exception();
                lock.lock();
                consumer_active_ = false;
                if (error_ == nullptr) {
                    error_ = callback_error;
                }
                mode_ = QueueMode::Failed;
                NotifyAllLocked();
                std::rethrow_exception(callback_error);
            }
            lock.lock();
            head_ = Next(head_);
            --size_;
            consumer_active_ = false;
            cv_not_full_.notify_all();
            NotifyDrainedIfLocked();
        }
        head_ = 0;
        tail_ = 0;
        mode_ = QueueMode::Stopped;
        NotifyAllLocked();
    }

    void Fail(std::exception_ptr error)
    {
        if (error == nullptr) {
            error = std::make_exception_ptr(std::runtime_error("ordered MPSC queue failed"));
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (error_ == nullptr) {
            error_ = std::move(error);
        }
        mode_ = QueueMode::Failed;
        NotifyAllLocked();
    }

    void RethrowIfFailed() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        RethrowIfFailedLocked();
    }

    QueueSnapshot Snapshot() const
    {
        std::lock_guard<std::mutex> lock(mu_);
        return QueueSnapshot{capacity_, size_, active_producers_, consumer_active_, mode_, error_ != nullptr};
    }

    template <typename Inspect>
    bool InspectHead(Inspect&& inspect) const
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (size_ == 0 || consumer_active_) {
            return false;
        }
        std::forward<Inspect>(inspect)(head_);
        return true;
    }

private:
    template <typename WaitHooks>
    static void WaitOnce(
        std::condition_variable& condition,
        std::unique_lock<std::mutex>& lock,
        WaitHooks& wait_hooks)
    {
        wait_hooks.BeforeWait();
        try {
            condition.wait(lock);
        } catch (...) {
            const std::exception_ptr wait_error = std::current_exception();
            if (lock.owns_lock()) {
                lock.unlock();
            }
            try {
                wait_hooks.AfterWait();
            } catch (...) {
                lock.lock();
                throw;
            }
            lock.lock();
            std::rethrow_exception(wait_error);
        }
        // Restoring an attached language-runtime state may take that
        // runtime's lock. Never invert its lock order with the queue mutex.
        lock.unlock();
        try {
            wait_hooks.AfterWait();
        } catch (...) {
            lock.lock();
            throw;
        }
        lock.lock();
    }

    uint32_t Next(uint32_t index) const noexcept
    {
        return static_cast<uint32_t>((static_cast<size_t>(index) + 1) % capacity_);
    }

    bool IsDrainedLocked() const noexcept
    {
        return size_ == 0 && active_producers_ == 0 && !consumer_active_;
    }

    void NotifyDrainedIfLocked()
    {
        if (IsDrainedLocked()) {
            cv_drained_.notify_all();
        }
    }

    void NotifyAllLocked()
    {
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
        cv_drained_.notify_all();
        cv_no_active_producers_.notify_all();
    }

    void RethrowIfFailedLocked() const
    {
        if (error_ != nullptr) {
            std::rethrow_exception(error_);
        }
    }

    const size_t capacity_;
    mutable std::mutex mu_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_drained_;
    std::condition_variable cv_no_active_producers_;
    uint32_t head_ = 0;
    uint32_t tail_ = 0;
    size_t size_ = 0;
    size_t active_producers_ = 0;
    bool consumer_active_ = false;
    QueueMode mode_ = QueueMode::Stopped;
    std::exception_ptr error_;
};

} // namespace c10_npu::queue
