#pragma once
#include <string>
#include <vector>
#include <c10/macros/Export.h>
#include <c10/util/Optional.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace torch_npu {
namespace unwind {
// gather current stack, relatively fast.
// gets faster once the cache of program counter locations is warm.
TORCH_NPU_API std::vector<void*> unwind();

struct Frame {
    std::string filename;
    std::string funcname;
    uint64_t lineno;
};

// note: symbolize is really slow
// it will launch an addr2line process that has to parse dwarf
// information from the libraries that frames point into.
// Callers should first batch up all the unique void* pointers
// across a number of unwind states and make a single call to
// symbolize.
TORCH_NPU_API std::vector<Frame> symbolize(const std::vector<void*>& frames);

// returns path to the library, and the offset of the addr inside the library
TORCH_NPU_API c10::optional<std::pair<std::string, uint64_t> > libraryFor(void* addr);

struct Stats {
    size_t hits = 0;
    size_t misses = 0;
    size_t unsupported = 0;
    size_t resets = 0;
};
Stats stats();

} // namespace unwind
} // namespace torch_npu
