#include "torch_npu/csrc/core/npu/ExperimentalSequenceMpscQueueCore.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using Core = c10_npu::queue::ExperimentalSequenceMpscQueueCore;
using Hooks = c10_npu::queue::NoopWaitHooks;
using Mode = c10_npu::queue::QueueMode;
using namespace std::chrono_literals;

[[noreturn]] void Fail(const std::string& message)
{
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::fflush(stderr);
    // A broken queue may have live waiters or a destructor that cannot finish.
    // Exit this owned CTest process rather than unwinding into an unbounded join.
    std::_Exit(1);
}

void Check(bool condition, const std::string& message)
{
    if (!condition) {
        Fail(message);
    }
}

class Event {
public:
    void Signal()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        signaled_ = true;
        changed_.notify_all();
    }

    bool IsSet()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return signaled_;
    }

    bool WaitFor(std::chrono::milliseconds timeout = 5000ms)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, timeout, [&] { return signaled_; });
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool signaled_ = false;
};

class ProcessDeadline {
public:
    ProcessDeadline() : thread_([this] {
        if (!finished_.WaitFor(15000ms)) {
            Fail("15-second internal process deadline exceeded");
        }
    })
    {}

    ~ProcessDeadline()
    {
        finished_.Signal();
        thread_.join();
    }

    ProcessDeadline(const ProcessDeadline&) = delete;
    ProcessDeadline& operator=(const ProcessDeadline&) = delete;

private:
    Event finished_;
    std::thread thread_;
};

class Worker {
public:
    explicit Worker(std::function<void()> function)
        : future_(done_.get_future()), thread_([this, function = std::move(function)] {
            try {
                function();
                done_.set_value();
            } catch (...) {
                done_.set_exception(std::current_exception());
            }
        })
    {}

    ~Worker()
    {
        if (thread_.joinable()) {
            Fail("test left an unjoined owned worker");
        }
    }

    bool Ready(std::chrono::milliseconds timeout = 0ms)
    {
        return future_.wait_for(timeout) == std::future_status::ready;
    }

    void Join(const std::string& label, std::chrono::milliseconds timeout = 5000ms)
    {
        Check(Ready(timeout), label + " did not finish before its deadline");
        try {
            future_.get();
        } catch (const std::exception& error) {
            Fail(label + " raised unexpectedly: " + error.what());
        } catch (...) {
            Fail(label + " raised a non-standard exception");
        }
        // Future readiness bounds queue work; the process watchdog also covers
        // thread epilogue and every join, including unexpected library failures.
        thread_.join();
    }

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

private:
    std::promise<void> done_;
    std::future<void> future_;
    std::thread thread_;
};

template <typename Predicate>
void Eventually(Predicate predicate, const std::string& label)
{
    const auto end = std::chrono::steady_clock::now() + 3000ms;
    while (!predicate()) {
        Check(std::chrono::steady_clock::now() < end, label + " did not become observable");
        std::this_thread::yield();
    }
}

template <typename Function>
void ExpectError(Function function, const std::string& message)
{
    try {
        function();
    } catch (const std::runtime_error& error) {
        Check(error.what() == message, "wrong canonical error: " + std::string(error.what()));
        return;
    } catch (...) {
        Fail("wrong exception type; expected runtime_error: " + message);
    }
    Fail("expected exception was not thrown: " + message);
}

void CheckStopped(Core& queue)
{
    const auto snapshot = queue.Snapshot();
    Check(snapshot.mode == Mode::Stopped && snapshot.size == 0 &&
              snapshot.active_producers == 0 && !snapshot.consumer_active,
          "stopped queue retained active producers, consumer, or items");
}

void HealthyCycle(Core& queue)
{
    Hooks hooks;
    queue.Start();
    Check(!queue.Snapshot().failed, "Start retained the previous failure");
    unsigned value = 0;
    Check(queue.Enqueue([&](uint32_t) { value = 7; }, hooks), "restart enqueue rejected");
    Check(queue.ConsumeOne([&](uint32_t) { Check(value == 7, "restart payload changed"); }),
          "restart consumer returned false");
    queue.StopAndDrain();
    CheckStopped(queue);
}

void CapacityRejected()
{
    for (size_t capacity : {size_t{0}, size_t{1}}) {
        bool rejected = false;
        try {
            Core queue(capacity);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        Check(rejected, "capacity " + std::to_string(capacity) + " must be rejected");
    }
}

struct BlockingAfterWait {
    Event before;
    Event after;
    Event release;
    void BeforeWait() { before.Signal(); }
    void AfterWait()
    {
        after.Signal();
        Check(release.WaitFor(), "test never released consumer AfterWait");
    }
};

struct ObserveWait {
    Event entered;
    void BeforeWait() { entered.Signal(); }
    void AfterWait() {}
};

void ClaimedLastDrain()
{
    Core queue(2);
    queue.Start();
    BlockingAfterWait consumer_hooks;
    std::atomic<unsigned> callbacks{0};
    Worker consumer([&] {
        Check(queue.ConsumeOne([&](uint32_t) { ++callbacks; }, consumer_hooks),
              "last-item consumer returned false");
    });
    Check(consumer_hooks.before.WaitFor(), "consumer did not enter empty wait");
    Hooks hooks;
    Check(queue.Enqueue([](uint32_t) {}, hooks), "last-item enqueue rejected");
    Check(consumer_hooks.after.WaitFor(), "consumer did not reach claimed AfterWait gate");
    Check(callbacks.load() == 0, "callback ran through the closed AfterWait gate");
    ObserveWait drain_hooks;
    Worker drain([&] { queue.WaitDrained(drain_hooks); });
    // This handshake observes the actual WaitDrained wait, or its erroneous
    // completion; scheduling a thread alone cannot make this regression pass.
    Eventually([&] { return drain.Ready() || drain_hooks.entered.IsSet(); }, "drain entry");
    Check(!drain.Ready(), "WaitDrained returned after last-item claim but before callback");
    Check(drain_hooks.entered.IsSet(), "WaitDrained did not observe the active consumer");
    Check(!drain.Ready(100ms), "WaitDrained returned while consumer AfterWait remained blocked");
    consumer_hooks.release.Signal();
    consumer.Join("claimed-last consumer");
    drain.Join("claimed-last drain");
    Check(callbacks.load() == 1, "last callback was lost or repeated");
    queue.StopAndDrain();
    CheckStopped(queue);
}

void FailedConsumerDrain()
{
    Core queue(2);
    queue.Start();
    Hooks hooks;
    Check(queue.Enqueue([](uint32_t) {}, hooks), "first enqueue rejected");
    Check(queue.Enqueue([](uint32_t) {}, hooks), "second enqueue rejected");
    Worker consumer([&] {
        ExpectError([&] {
            queue.ConsumeOne([](uint32_t) { throw std::runtime_error("first consumer failed"); });
        }, "first consumer failed");
    });
    consumer.Join("throwing first consumer");
    Check(queue.Snapshot().size == 1, "test requires a pending item behind the failed callback");
    Worker drain([&] { ExpectError([&] { queue.WaitDrained(hooks); }, "first consumer failed"); });
    drain.Join("failed consumer drain with pending item", 2000ms);
    queue.StopAndDiscardMetadata();
    CheckStopped(queue);
    HealthyCycle(queue);
}

void InspectGateConsumer()
{
    Core queue(2);
    queue.Start();
    Hooks hooks;
    unsigned payload = 0;
    Check(queue.Enqueue([&](uint32_t) { payload = 11; }, hooks), "inspect test enqueue rejected");
    Event inspecting;
    Event release_inspector;
    Worker inspector([&] {
        Check(queue.InspectHead([&](uint32_t) {
            Check(payload == 11, "inspected payload differs");
            inspecting.Signal();
            Check(release_inspector.WaitFor(), "inspector was never released");
        }), "published head was not inspectable");
    });
    Check(inspecting.WaitFor(), "inspector did not acquire the head gate");
    ObserveWait consumer_hooks;
    std::atomic<unsigned> callbacks{0};
    Worker consumer([&] {
        Check(queue.ConsumeOne([&](uint32_t) {
            Check(payload == 11, "payload changed across inspector/consumer handoff");
            ++callbacks;
        }, consumer_hooks), "consumer returned false after inspection");
    });
    Eventually([&] { return consumer.Ready() || consumer_hooks.entered.IsSet(); }, "consumer gate wait");
    Check(!consumer.Ready(), "consumer returned/threw while a legitimate InspectHead held its gate");
    Check(consumer_hooks.entered.IsSet(), "consumer did not wait for the inspector gate");
    Check(callbacks.load() == 0, "consumer read payload through an active inspector");
    Check(!consumer.Ready(100ms), "consumer completed before inspector release");
    release_inspector.Signal();
    inspector.Join("inspector");
    consumer.Join("consumer waiting behind inspector");
    Check(callbacks.load() == 1, "inspection handoff lost or repeated consumption");
    queue.StopAndDrain();
    CheckStopped(queue);
}

struct ThrowWait {
    explicit ThrowWait(bool after) : throw_after(after) {}
    bool throw_after;
    Event entered;
    std::atomic<unsigned> before_calls{0};
    std::atomic<unsigned> after_calls{0};
    void BeforeWait()
    {
        ++before_calls;
        entered.Signal();
        if (!throw_after) {
            throw std::runtime_error("before-wait failed");
        }
    }
    void AfterWait()
    {
        ++after_calls;
        throw std::runtime_error("after-wait failed");
    }
};

void ProducerHookThrows(bool after)
{
    Core queue(2);
    queue.Start();
    Hooks hooks;
    std::array<unsigned, 2> slots{};
    for (unsigned value = 0; value != 2; ++value) {
        Check(queue.Enqueue([&](uint32_t slot) { slots[slot] = value; }, hooks), "fill rejected");
    }
    ThrowWait throwing(after);
    std::atomic<unsigned> copies{0};
    Worker producer([&] {
        ExpectError([&] { queue.Enqueue([&](uint32_t) { ++copies; }, throwing); },
                    after ? "after-wait failed" : "before-wait failed");
    });
    Check(throwing.entered.WaitFor(), "producer never reached a full-queue wait");
    if (after) {
        Check(queue.ConsumeOne([&](uint32_t slot) { Check(slots[slot] == 0, "wrong first payload"); }),
              "freeing first slot returned false");
    }
    producer.Join("throwing producer wait hook", 2000ms);
    const auto snapshot = queue.Snapshot();
    Check(snapshot.active_producers == 0, "throwing producer hook leaked admission");
    Check(copies.load() == 0, "copy callback ran after a throwing wait hook");
    Check(throwing.before_calls.load() == 1 && throwing.after_calls.load() == unsigned(after),
          "throwing wait hook was retried or restored without a successful BeforeWait");
    if (snapshot.failed) {
        // A core may publish a failed tombstone after reservation, or restore
        // hooks before reserving. Either path must close admission and cleanup.
        ExpectError([&] { queue.RethrowIfFailed(); }, after ? "after-wait failed" : "before-wait failed");
        Worker cleanup([&] { queue.StopAndDiscardMetadata(); });
        cleanup.Join("failed-hook metadata cleanup", 2000ms);
    } else {
        const unsigned first = after ? 1 : 0;
        for (unsigned value = first; value != 2; ++value) {
            Check(queue.ConsumeOne([&](uint32_t slot) { Check(slots[slot] == value, "pending payload changed"); }),
                  "pending item disappeared after hook failure");
        }
        Worker drain([&] { queue.WaitDrained(hooks); });
        drain.Join("healthy-hook reservation drain", 2000ms);
        queue.StopAndDrain();
    }
    CheckStopped(queue);
    HealthyCycle(queue);
}

void ConsumerHookThrows(bool after)
{
    Core queue(2);
    queue.Start();
    ThrowWait throwing(after);
    std::atomic<unsigned> callbacks{0};
    Worker consumer([&] {
        ExpectError([&] { queue.ConsumeOne([&](uint32_t) { ++callbacks; }, throwing); },
                    after ? "after-wait failed" : "before-wait failed");
    });
    Check(throwing.entered.WaitFor(), "consumer never reached empty wait");
    Hooks hooks;
    if (after) {
        Check(queue.Enqueue([](uint32_t) {}, hooks), "wake consumer enqueue rejected");
    }
    consumer.Join("throwing consumer wait hook", 2000ms);
    Check(callbacks.load() == 0, "consumer callback ran after hook failure");
    const auto snapshot = queue.Snapshot();
    Check(snapshot.active_producers == 0 && !snapshot.consumer_active,
          "throwing consumer hook leaked ownership");
    Check(throwing.before_calls.load() == 1 && throwing.after_calls.load() == unsigned(after),
          "consumer wait hook was retried");
    if (snapshot.failed) {
        ExpectError([&] { queue.RethrowIfFailed(); }, after ? "after-wait failed" : "before-wait failed");
        Worker cleanup([&] { queue.StopAndDiscardMetadata(); });
        cleanup.Join("consumer-hook cleanup", 2000ms);
    } else {
        // A claimed slot must have been returned even when AfterWait throws.
        std::array<unsigned, 2> slots{};
        Worker reuse([&] {
            for (unsigned value = 0; value != 2; ++value) {
                Check(queue.Enqueue([&](uint32_t slot) { slots[slot] = value; }, hooks), "slot reuse rejected");
            }
        });
        reuse.Join("consumer-hook slot reuse", 2000ms);
        for (unsigned value = 0; value != 2; ++value) {
            Check(queue.ConsumeOne([&](uint32_t slot) { Check(slots[slot] == value, "reused slot payload changed"); }),
                  "reused slot could not be consumed");
        }
        Worker drain([&] { queue.WaitDrained(hooks); });
        drain.Join("consumer-hook drain", 2000ms);
        queue.StopAndDrain();
    }
    CheckStopped(queue);
    HealthyCycle(queue);
}

void StopFull()
{
    Core queue(2);
    queue.Start();
    Hooks hooks;
    std::array<unsigned, 2> slots{};
    for (unsigned value = 0; value != 2; ++value) {
        Check(queue.Enqueue([&](uint32_t slot) { slots[slot] = value; }, hooks), "fill rejected");
    }
    std::array<ObserveWait, 4> waits;
    std::atomic<unsigned> copies{0};
    std::vector<std::unique_ptr<Worker>> producers;
    for (size_t i = 0; i != waits.size(); ++i) {
        producers.emplace_back(new Worker([&, i] {
            Check(!queue.Enqueue([&](uint32_t) { ++copies; }, waits[i]), "stopped full waiter published");
        }));
    }
    for (auto& wait : waits) {
        Check(wait.entered.WaitFor(), "full producer did not enter wait");
    }
    queue.RequestStop();
    for (auto& producer : producers) {
        producer->Join("stopped full producer", 2000ms);
    }
    Check(copies.load() == 0 && queue.Snapshot().active_producers == 0, "stop leaked a producer/copy");
    unsigned discarded = 0;
    queue.StopAndDiscard([&](uint32_t slot) { Check(slots[slot] == discarded++, "discard order changed"); });
    Check(discarded == 2, "stop lost or duplicated pending items");
    CheckStopped(queue);
    Check(!queue.Enqueue([](uint32_t) {}, hooks), "post-stop enqueue accepted");
    HealthyCycle(queue);
}

void CopyThrows()
{
    Core queue(2);
    queue.Start();
    Hooks hooks;
    ExpectError([&] { queue.Enqueue([](uint32_t) { throw std::runtime_error("copy failed"); }, hooks); },
                "copy failed");
    Check(queue.Snapshot().active_producers == 0, "copy failure leaked producer");
    unsigned callbacks = 0;
    ExpectError([&] { queue.ConsumeOne([&](uint32_t) { ++callbacks; }); }, "copy failed");
    Check(callbacks == 0, "aborted copy reached consumer callback");
    queue.StopAndDiscardMetadata();
    CheckStopped(queue);
    HealthyCycle(queue);
}

void Discard(bool throwing)
{
    Core queue(8);
    queue.Start();
    Hooks hooks;
    std::array<std::string, 8> slots;
    for (unsigned value = 0; value != slots.size(); ++value) {
        Check(queue.Enqueue([&](uint32_t slot) { slots[slot] = std::to_string(value); }, hooks), "discard fill rejected");
    }
    unsigned count = 0;
    auto discard = [&](uint32_t slot) {
        Check(slots[slot] == std::to_string(count), "discard lost FIFO or repeated payload");
        slots[slot].clear();
        ++count;
        if (throwing) {
            throw std::runtime_error("discard failed");
        }
    };
    if (throwing) {
        ExpectError([&] { queue.StopAndDiscard(discard); }, "discard failed");
        Check(count == 1, "throwing discard callback was retried");
        queue.StopAndDiscardMetadata();
    } else {
        queue.StopAndDiscard(discard);
        Check(count == slots.size(), "discard item count differs");
    }
    CheckStopped(queue);
    HealthyCycle(queue);
}

void Restart()
{
    Core queue(8);
    for (unsigned cycle = 0; cycle != 100; ++cycle) {
        HealthyCycle(queue);
    }
}

struct Payload {
    unsigned producer = 0;
    unsigned sequence = 0;
    uint64_t value = 0;
    std::array<uint64_t, 8> fixed{};
    std::string text;
    std::vector<unsigned> owned;
};

Payload MakePayload(unsigned producer, unsigned sequence, bool owning)
{
    Payload data;
    data.producer = producer;
    data.sequence = sequence;
    data.value = (uint64_t(producer) << 32) | sequence;
    data.fixed.fill(data.value ^ UINT64_C(0xabc0123456789));
    if (owning) {
        data.text = std::to_string(data.value);
        data.owned = {producer, sequence, sequence ^ 0x1234U};
    }
    return data;
}

void Stress(size_t capacity)
{
    constexpr unsigned producer_count = 4;
    constexpr unsigned items = 2000;
    Core queue(capacity);
    std::vector<Payload> slots(capacity);
    for (bool owning : {false, true}) {
        for (unsigned cycle = 0; cycle != 2; ++cycle) {
            queue.Start();
            std::array<unsigned, producer_count> next{};
            unsigned consumed = 0;
            Worker consumer([&] {
                while (consumed != producer_count * items) {
                    Check(queue.ConsumeOne([&](uint32_t slot) {
                        const auto& data = slots[slot];
                        Check(data.producer < producer_count && data.sequence == next[data.producer],
                              "per-producer FIFO/exactly-once delivery failed");
                        const auto expected = MakePayload(data.producer, data.sequence, owning);
                        Check(data.value == expected.value && data.fixed == expected.fixed &&
                                  data.text == expected.text && data.owned == expected.owned,
                              "fixed/owning payload changed before consumption");
                        ++next[data.producer];
                        ++consumed;
                    }), "stress consumer stopped early");
                }
            });
            std::vector<std::unique_ptr<Worker>> producers;
            for (unsigned producer = 0; producer != producer_count; ++producer) {
                producers.emplace_back(new Worker([&, producer] {
                    Hooks hooks;
                    for (unsigned sequence = 0; sequence != items; ++sequence) {
                        const auto data = MakePayload(producer, sequence, owning);
                        Check(queue.Enqueue([&](uint32_t slot) { slots[slot] = data; }, hooks),
                              "stress producer stopped early");
                    }
                }));
            }
            for (auto& producer : producers) {
                producer->Join("stress producer");
            }
            consumer.Join("stress consumer");
            for (unsigned count : next) {
                Check(count == items, "producer item count differs");
            }
            Worker drain([&] { Hooks hooks; queue.WaitDrained(hooks); });
            drain.Join("stress drain");
            queue.StopAndDrain();
            CheckStopped(queue);
        }
    }
}
} // namespace

int main(int argc, char** argv)
{
    ProcessDeadline deadline;
    if (argc != 2) {
        Fail("one case name required");
    }
    const std::string name = argv[1];
    try {
        if (name == "capacity-rejected") CapacityRejected();
        else if (name == "claimed-last-drain") ClaimedLastDrain();
        else if (name == "inspect-gate-consumer") InspectGateConsumer();
        else if (name == "failed-consumer-drain") FailedConsumerDrain();
        else if (name == "producer-before-wait-throws") ProducerHookThrows(false);
        else if (name == "producer-after-wait-throws") ProducerHookThrows(true);
        else if (name == "consumer-before-wait-throws") ConsumerHookThrows(false);
        else if (name == "consumer-after-wait-throws") ConsumerHookThrows(true);
        else if (name == "stop-full") StopFull();
        else if (name == "copy-throws") CopyThrows();
        else if (name == "discard") Discard(false);
        else if (name == "discard-throws") Discard(true);
        else if (name == "restart") Restart();
        else if (name == "stress-2") Stress(2);
        else if (name == "stress-8") Stress(8);
        else if (name == "stress-64") Stress(64);
        else if (name == "stress-4096") Stress(4096);
        else Fail("unknown test case: " + name);
    } catch (const std::exception& error) {
        Fail("uncaught test exception: " + std::string(error.what()));
    } catch (...) {
        Fail("uncaught non-standard test exception");
    }
    std::cout << "PASS " << name << '\n';
    return 0;
}
