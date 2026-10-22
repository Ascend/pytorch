#pragma once

#if __has_include(<c10/core/RingBuffer.h>)
#include <c10/core/RingBuffer.h>
#else
// PyTorch 2.13 does not provide c10/core/RingBuffer.h. Keep this fallback in
// sync with the upstream implementation; newer PyTorch uses its own header.
// Source: pytorch/pytorch@8cd072a84066f964d0fd62647f0155f8344a4f18
// c10/core/RingBuffer.h (BSD-3-Clause; see LICENSE).
#include <c10/macros/Macros.h>

#include <algorithm>
#include <mutex>
#include <vector>

#if C10_ASAN_ENABLED
#include <sanitizer/lsan_interface.h>
#endif

namespace c10 {

template <class T>
class RingBuffer {
 public:
  RingBuffer() {
    // alloc_trace is a heap pointer we intentionally never free: it can hold
    // references to Python state that is already destroyed by the time exit
    // handlers run, so freeing it then is unsafe. The owning allocator impl is
    // itself heap-allocated and destroyed at exit, which makes this allocation
    // unreachable and thus flagged by LeakSanitizer; annotate it as a
    // deliberate leak so LSan ignores it (and everything it transitively holds,
    // e.g. recorded Python contexts).
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    alloc_trace = new std::vector<T>();
#if C10_ASAN_ENABLED
    __lsan_ignore_object(alloc_trace);
#endif
  }

  void setMaxEntries(size_t size) {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    alloc_trace_max_entries_ = std::max(static_cast<size_t>(1), size);
  }

  void insertEntries(const T& entry) {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    if (alloc_trace->size() < alloc_trace_max_entries_) {
      alloc_trace->emplace_back(entry);
    } else {
      (*alloc_trace)[alloc_trace_next++] = entry;
      if (alloc_trace_next == alloc_trace_max_entries_) {
        alloc_trace_next = 0;
      }
    }
  }

  void getEntries(std::vector<T>& result) const {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    result.reserve(result.size() + alloc_trace->size());
    std::rotate_copy(
        alloc_trace->begin(),
        std::next(alloc_trace->begin(), alloc_trace_next),
        alloc_trace->end(),
        std::back_inserter(result));
  }

  void clear() {
    std::lock_guard<std::mutex> lk(alloc_trace_lock);
    alloc_trace_next = 0;
    alloc_trace->clear();
    alloc_trace->shrink_to_fit();
  }

 private:
  size_t alloc_trace_max_entries_ = 1;

  mutable std::mutex alloc_trace_lock;
  size_t alloc_trace_next = 0;
  std::vector<T>* alloc_trace;
};

} // namespace c10
#endif
