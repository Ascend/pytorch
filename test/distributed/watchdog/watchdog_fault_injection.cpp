// Watchdog integration test LD_PRELOAD library. No production injection switches.
// Includes dump failures exercised through the watchdog heartbeat monitor.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <ios>
#include <functional>
#include <optional>
#include <pthread.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include <hccl/hccl.h>
#include <torch/csrc/profiler/combined_traceback.h>

namespace {
std::atomic<int> mode{0};
std::atomic<bool> firstStack{true};
std::atomic<bool> firstWrite{true};
std::atomic<int> serializedDumps{0};
std::atomic<void*> traceBuffer{nullptr};
std::atomic<bool> failWriter{false};
std::atomic<bool> releaseStalls{false};
std::atomic<bool> releaseGilCheck{false};
std::atomic<bool> firstSync{true};
using GilChecker = bool (*)();
GilChecker originalGilChecker = nullptr;
std::string markerPath;
std::string writerTarget;

void printCrashBacktrace(int signal) {
  void* frames[48];
  const int count = backtrace(frames, 48);
  backtrace_symbols_fd(frames, count, STDERR_FILENO);
  ::signal(signal, SIG_DFL);
  ::raise(signal);
}

void mark(const char* event) {
  const auto seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  char line[160];
  const int length = std::snprintf(line, sizeof(line), "%.6f %s\n", seconds, event);
  const int fd = open(markerPath.c_str(), O_CREAT | O_APPEND | O_WRONLY, 0600);
  if (fd >= 0) {
    (void)write(fd, line, length);
    close(fd);
  }
}

template <typename T>
T realFunction(const char* name, const char* library = nullptr) {
  void* result = dlsym(RTLD_NEXT, name);
  if (!result && library) {
    void* handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (handle) {
      result = dlsym(handle, name);
    }
  }
  if (!result) {
    std::fprintf(stderr, "Watchdog test cannot resolve %s: %s\n", name, dlerror());
    std::abort();
  }
  return reinterpret_cast<T>(result);
}

void stallUntilReleased() {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (!releaseStalls.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

bool isWatchdogThread() {
  char name[16] = {};
  pthread_getname_np(pthread_self(), name, sizeof(name));
  return std::string(name).find("hccl_watch") == 0;
}

bool calledFromCommPrecheck() {
  void* frames[32];
  const int count = backtrace(frames, 32);
  for (int i = 0; i < count; ++i) {
    Dl_info info{};
    if (dladdr(frames[i], &info) && info.dli_sname &&
        std::string(info.dli_sname).find("checkHcclComms") != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool controlledGilCheck() {
  mark("gil_check_entered");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!releaseGilCheck.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  mark("gil_acquire_started");
  const bool acquired = originalGilChecker();
  mark(acquired ? "gil_acquired" : "gil_uninitialized");
  return acquired;
}
} // namespace

// Return setup failures to ctypes; C++ exceptions must not cross the Python FFI.
extern "C" int watchdog_test_arm(int value, const char* path) noexcept {
  mode.store(0);
  try {
    markerPath = path;
    signal(SIGSEGV, printCrashBacktrace);
    if (value == 20) {
      using Dumper = std::function<void(std::function<void(const std::string&)>)>;
      using Getter = std::optional<Dumper>& (*)();
      auto getDumper = realFunction<Getter>(
          "_ZN8c10d_npu20get_cpp_trace_dumperB5cxx11Ev", "libtorch_npu.so");
      getDumper() = [](std::function<void(const std::string&)> writeLine) {
        mark("cpp_dump_called");
        writeLine("watchdog test C++ stack callback");
      };
    }
    if (value == 21) {
      using Getter = GilChecker& (*)();
      auto getChecker = realFunction<Getter>(
          "_ZN8c10d_npu15get_gil_checkerEv", "libtorch_npu.so");
      originalGilChecker = getChecker();
      if (!originalGilChecker) {
        std::fprintf(stderr, "Python GIL checker is not registered\n");
        return 1;
      }
      getChecker() = &controlledGilCheck;
    }
    mode.store(value);
    mark("armed");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Watchdog fault injection setup failed: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "Watchdog fault injection setup failed with an unknown exception\n");
  }
  mode.store(0);
  return 2;
}

extern "C" void watchdog_test_release_stalls() { releaseStalls.store(true); }

extern "C" void watchdog_test_disarm() {
  mode.store(0);
  failWriter.store(false);
  releaseStalls.store(true);
}

extern "C" void watchdog_test_marker(const char* event) { mark(event); }

extern "C" void watchdog_test_set_writer_target(const char* path) { writerTarget = path; }

// Call through ctypes.PyDLL: it holds the GIL while the native checker tries to acquire it.
extern "C" void watchdog_test_hold_gil(int milliseconds) {
  releaseGilCheck.store(true);
  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

extern "C" void watchdog_test_fail_writer() {
  failWriter.store(true);
}

extern "C" HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult* error) {
  if (mode.load() == 10 && isWatchdogThread()) {
    if (calledFromCommPrecheck()) {
      *error = HCCL_SUCCESS;
      return HCCL_SUCCESS;
    }
    mark("work_async_error");
    *error = HCCL_E_REMOTE;
    return HCCL_SUCCESS;
  }
  using Fn = HcclResult (*)(HcclComm, HcclResult*);
  static auto real = realFunction<Fn>("HcclGetCommAsyncError", "libhccl.so");
  return real(comm, error);
}

extern "C" HcclResult HcclCommDestroy(HcclComm comm) {
  if (mode.load() != 0) {
    mark("comm_destroy");
  }
  using Fn = HcclResult (*)(HcclComm);
  static auto real = realFunction<Fn>("HcclCommDestroy", "libhccl.so");
  return real(comm);
}

// Use ABI-compatible pointers so this shim also builds with older ACL headers.
int interceptEventQuery(void*, int*) asm("aclrtQueryEventStatus");
int interceptEventQuery(void* event, int* status) {
  const int current = mode.load();
  if (current == 40 && isWatchdogThread()) {
    mark("event_query_held");
    *status = 0; // ACL_EVENT_RECORDED_STATUS_NOT_READY
    return 0;
  }
  if ((current >= 11 && current <= 14) || (current == 15 && isWatchdogThread())) {
    mark("event_query_error");
    switch (current) {
      case 12:
      case 15: throw std::runtime_error("injected driver shutting down");
      case 13: throw std::runtime_error("injected FORCE STOP");
      case 14: throw std::runtime_error("injected UCE ERROR");
      default: throw std::runtime_error("injected event query failure");
    }
  }
  using Fn = int (*)(void*, int*);
  static auto real = realFunction<Fn>("aclrtQueryEventStatus", "libascendcl.so");
  return real(event, status);
}

int interceptStreamSync(void*, int32_t) asm("aclrtSynchronizeStreamWithTimeout");
int interceptStreamSync(void* stream, int32_t timeout) {
  using Fn = int (*)(void*, int32_t);
  static auto real = realFunction<Fn>("aclrtSynchronizeStreamWithTimeout", "libascendcl.so");
  const auto result = real(stream, timeout);
  if (result == 0 && mode.load() == 30 && firstSync.exchange(false)) {
    mark("shutdown_sync_error");
    throw std::runtime_error("injected shutdown stream synchronization failure");
  }
  return result;
}

// libtorch_npu uses -Bsymbolic-functions, so inject at its external libtorch
// symbolization call rather than pretending a local HCCL call is interposable.
torch::SymbolizedTracebacks torch::symbolize(
    const std::vector<torch::CapturedTraceback*>& traces) {
  // The watchdog prints one failed-op traceback separately. A full dump in
  // these tests contains warmup + failing op, so inject only into that dump.
  if ((mode.load() == 1 || mode.load() == 2) && traces.size() >= 2) {
    mark("dump_with_stack");
    if (mode.load() == 1 && firstStack.exchange(false)) {
      mark("stack_stalled");
      stallUntilReleased();
    }
  }
  using Fn = torch::SymbolizedTracebacks (*)(
      const std::vector<torch::CapturedTraceback*>&);
  static auto real = realFunction<Fn>(
      "_ZN5torch9symbolizeERKSt6vectorIPNS_17CapturedTracebackESaIS2_EE",
      "libtorch_cpu.so");
  return real(traces);
}

void interceptPickleStop(void*) asm("_ZN5torch3jit7Pickler4stopEv");
void interceptPickleStop(void* self) {
  using Fn = void (*)(void*);
  static auto real = realFunction<Fn>("_ZN5torch3jit7Pickler4stopEv", "libtorch_cpu.so");
  real(self);
  if (mode.load() == 1 || mode.load() == 2) {
    const int previous = serializedDumps.fetch_add(1);
    mark(mode.load() == 1 || previous > 0
             ? "dump_without_stack" : "dump_serialized");
  } else if (mode.load() >= 4) {
    mark("dump_serialized");
  }
}

// The HCCL writer now implements file I/O locally. Intercept its external
// libstdc++ file calls; -Bsymbolic-functions binds the HCCL method internally.
void* interceptFileOpen(void*, const char*, std::ios_base::openmode)
    asm("_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode");
void* interceptFileOpen(void* self, const char* path, std::ios_base::openmode flags) {
  const bool active = mode.load() != 0;
  const std::string target = !active ? "" : (writerTarget.empty()
      ? markerPath.substr(0, markerPath.rfind('/') + 1) + "trace_0"
      : writerTarget);
  const bool isTrace = active && std::string(path) == target;
  if (isTrace && failWriter.load()) {
    mark("writer_threw");
    throw std::runtime_error("injected FR writer failure");
  }
  if (isTrace && mode.load() == 2 && firstWrite.exchange(false)) {
    mark("writer_stalled");
    stallUntilReleased();
  }
  using Fn = void* (*)(void*, const char*, std::ios_base::openmode);
  static auto real = realFunction<Fn>(
      "_ZNSt13basic_filebufIcSt11char_traitsIcEE4openEPKcSt13_Ios_Openmode",
      "libstdc++.so.6");
  void* result = real(self, path, flags);
  if (isTrace && result) {
    traceBuffer.store(self);
  }
  return result;
}

void* interceptFileClose(void*) asm("_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv");
void* interceptFileClose(void* self) {
  using Fn = void* (*)(void*);
  static auto real = realFunction<Fn>(
      "_ZNSt13basic_filebufIcSt11char_traitsIcEE5closeEv", "libstdc++.so.6");
  void* result = real(self);
  if (traceBuffer.load() == self) {
    traceBuffer.store(nullptr);
    mark("write_completed");
  }
  return result;
}
