// Test-only LD_PRELOAD library. No injection switch is added to production code.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <ios>
#include <functional>
#include <optional>
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
std::string markerPath;

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
    std::fprintf(stderr, "FR test cannot resolve %s: %s\n", name, dlerror());
    std::abort();
  }
  return reinterpret_cast<T>(result);
}
} // namespace

extern "C" void fr_test_arm(int value, const char* path) {
  markerPath = path;
  signal(SIGSEGV, printCrashBacktrace);
  mode.store(value);
  mark("armed");
  if (value == 6 || value == 7) {
    // Hold the monitor after its FR phase while the watchdog finishes its
    // existing error grace. This exposes reissuing a consumed dump request.
    using Dumper = std::function<void(std::function<void(const std::string&)>)>;
    using Getter = std::optional<Dumper>& (*)();
    auto getDumper = realFunction<Getter>(
        "_ZN8c10d_npu20get_cpp_trace_dumperB5cxx11Ev", "libtorch_npu.so");
    getDumper() = [value](std::function<void(const std::string&)>) {
      mark(value == 6 ? "monitor_after_dump" : "monitor_exit_diagnostics");
      if (value == 6) {
        std::this_thread::sleep_for(std::chrono::seconds(6));
      }
    };
  }
}

extern "C" void fr_test_fail_writer() {
  failWriter.store(true);
}

extern "C" HcclResult HcclAllReduce(
    void* input, void* output, uint64_t count, HcclDataType type,
    HcclReduceOp op, HcclComm comm, aclrtStream stream) {
  if (mode.load() == 3) {
    mark("sync_api_error");
    return HCCL_E_INTERNAL;
  }
  using Fn = HcclResult (*)(void*, void*, uint64_t, HcclDataType,
                            HcclReduceOp, HcclComm, aclrtStream);
  static auto real = realFunction<Fn>("HcclAllReduce", "libhccl.so");
  return real(input, output, count, type, op, comm, stream);
}

extern "C" HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult* error) {
  if (mode.load() == 4) {
    mark("async_comm_error");
    *error = HCCL_E_REMOTE;
    return HCCL_SUCCESS;
  }
  if (mode.load() == 5) {
    mark("async_query_api_error");
    return HCCL_E_INTERNAL;
  }
  using Fn = HcclResult (*)(HcclComm, HcclResult*);
  static auto real = realFunction<Fn>("HcclGetCommAsyncError", "libhccl.so");
  return real(comm, error);
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
      std::this_thread::sleep_for(std::chrono::seconds(60));
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
  const bool isTrace = mode.load() != 0 &&
      std::string(path) == markerPath.substr(0, markerPath.rfind('/') + 1) + "trace_0";
  if (isTrace && failWriter.load()) {
    mark("writer_threw");
    throw std::runtime_error("injected FR writer failure");
  }
  if (isTrace && mode.load() == 2 && firstWrite.exchange(false)) {
    mark("writer_stalled");
    std::this_thread::sleep_for(std::chrono::seconds(60));
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
