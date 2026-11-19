# Standalone sequence queue correctness tests

These tests require only the two queue headers, a C++17 compiler, CMake and
threads. They do not link Torch, NPU/CANN libraries or gtest. Every check remains
active under NDEBUG. This is correctness testing, not performance evidence.

```sh
cmake -S test/cpp/queue -B build/queue-tests -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/queue-tests --parallel 2
ctest --test-dir build/queue-tests --output-on-failure --timeout 20
```

For a historical RED use a different fresh build directory and
`-DQUEUE_SOURCE_ROOT=/path/to/frozen/header/root`. That root must contain
`torch_npu/csrc/core/npu/{ExperimentalSequenceMpscQueueCore.h,OrderedMpscQueueCore.h}`.
Keep the same test files for baseline and candidate. No historical baseline or
candidate execution is claimed by this source delivery.

Each CTest is a separate process. Futures/events bound thread coordination and
completion; an internal 15-second process watchdog and CTest's 20-second timeout
cover broken queue calls, destructors and joins. Failure terminates the owned test
process instead of unwinding through live queue waiters or detaching them.

The suite checks capacity 0/1 rejection, a last-item claim blocked in AfterWait
without an early WaitDrained return, prompt canonical failure with pending work,
and a consumer waiting behind a deterministic, blocked InspectHead callback,
throwing producer/consumer wait hooks, stop-full admission cleanup, copy failures,
FIFO discard and discard failure, restart, and four-producer fixed/owning payload
delivery at capacities 2/8/64/4096. The claim-gap test observes WaitDrained entering
its actual wait or completing; it does not infer correctness from thread start or
sleep alone. Hook-failure tests accept either safe pre-reservation restoration or
a canonical failed tombstone, while requiring no admission, reservation or slot
leak and successful bounded cleanup/restart.
