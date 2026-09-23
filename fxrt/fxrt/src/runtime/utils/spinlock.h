#ifndef __RUNTIME_UTILS_SPINLOCK_H__
#define __RUNTIME_UTILS_SPINLOCK_H__

#include <atomic>

namespace fxrt {
namespace runtime {
class SpinLock {
 public:
  void lock() {
    while (locked_.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() {
    locked_.clear(std::memory_order_release);
  }

 private:
  std::atomic_flag locked_ = ATOMIC_FLAG_INIT;
};
} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_UTILS_SPINLOCK_H__
