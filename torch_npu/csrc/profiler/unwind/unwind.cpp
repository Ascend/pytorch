#include <dlfcn.h>
#include <cxxabi.h>
#include <execinfo.h>
#include <iostream>

#include <c10/util/Exception.h>
#include <torch/csrc/utils/cpp_stacktraces.h>
#include "unwind.h"

#if defined(__linux__) && defined(__aarch64__)
#include <link.h>
#endif

#if !defined(__linux__) || !defined(__x86_64__) || !defined(__has_include) || !__has_include("ext/stdio_filebuf.h")
namespace torch_npu::unwind {
#if defined(__linux__) && defined(__aarch64__)
namespace {
// PyTorch's symbolizer reads a four-byte header from each EH segment.
bool hasInvalidEhSegment() {
  return dl_iterate_phdr(
             [](struct dl_phdr_info* info, size_t, void*) {
               for (size_t i = 0; i < info->dlpi_phnum; ++i) {
                 const auto& segment = info->dlpi_phdr[i];
                 if (segment.p_type == PT_GNU_EH_FRAME && segment.p_memsz < 4) {
                   return 1;
                 }
               }
               return 0;
             },
             nullptr) != 0;
}
} // namespace
#endif

std::vector<void*> unwind() {
#if defined(__linux__) && defined(__aarch64__)
  return torch::unwind::unwind();
#else
  const int size = 200;
  void* buffer[size];
  int nptrs = backtrace(buffer, size);
  return std::vector<void*>(buffer, buffer + nptrs);
#endif
}

c10::optional<std::pair<std::string, uint64_t>> libraryFor(void* addr) {
  TORCH_CHECK(false, "record_context_cpp is not support on non-linux non-x86_64 platforms");
}

std::vector<torch::unwind::Frame> symbolize(const std::vector<void*>& frames) {
#if defined(__linux__) && defined(__aarch64__)
  // Fall back to dladdr below when a loaded ELF object has an undersized EH segment.
  if (!hasInvalidEhSegment()) {
    return torch::unwind::symbolize(frames, torch::get_symbolize_mode());
  }
#endif
  std::vector<torch::unwind::Frame> results;
  for (const auto& addr : frames) {
    torch::unwind::Frame frame;
    Dl_info info;
    frame.lineno = 0;
    if (dladdr(addr, &info)) {
      frame.filename = info.dli_fname ? info.dli_fname : "??";
      size_t last_pos = frame.filename.find_last_of('/');
      if (last_pos != std::string::npos) {
        frame.filename = frame.filename.substr(last_pos + 1);
      }
      char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, nullptr);
      if (demangled) {
        frame.funcname = demangled;
        free(demangled);
        demangled = nullptr;
      } else {
        frame.funcname = info.dli_sname ? info.dli_sname : "??";
      }
    } else {
      frame.filename = "??";
      frame.funcname = "??";
    }
    if ((frame.filename == "python" && frame.filename.find("PyEval_EvalFrame") == std::string::npos) ||
        (frame.filename.find("libc.so") != std::string::npos)) {
      frame.funcname = "__libc_start_main";
    }
    results.push_back(frame);
  }
  return results;
}

Stats stats() {
  TORCH_CHECK(false, "record_context_cpp is not support on non-linux non-x86_64 platforms");
}

} // namespace torch_npu::unwind

#else

#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <unistd.h>
#include <vector>

namespace torch_npu::unwind {
std::vector<void*> unwind() {
  TORCH_CHECK(false, "For the linux x86 platform, this function should call the torch function");
}

c10::optional<std::pair<std::string, uint64_t>> libraryFor(void* addr) {
  TORCH_CHECK(false, "For the linux x86 platform, this function should call the torch function");
}

std::vector<torch::unwind::Frame> symbolize(const std::vector<void*>& frames) {
  TORCH_CHECK(false, "For the linux x86 platform, this function should call the torch function");
}

Stats stats() {
  TORCH_CHECK(false, "For the linux x86 platform, this function should call the torch function");
}

} // namespace torch_npu::unwind
#endif
