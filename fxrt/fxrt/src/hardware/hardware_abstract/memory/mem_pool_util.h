#ifndef FXRT_SRC_HARDWARE_MEMORY_MEM_POOL_UTIL_H_
#define FXRT_SRC_HARDWARE_MEMORY_MEM_POOL_UTIL_H_

#include <atomic>
#include <string>

#include "common/visible.h"

namespace fxrt {
namespace memory {
namespace mem_pool {
enum class MemType : int {
  kWeight = 0,
  kConstantValue,
  kKernel,
  kGraphOutput,
  kSomas,
  kSomasOutput,
  kGeConst,
  kGeFixed,
  kBatchMemory,
  kContinuousMemory,
  kPyNativeInput = 10,
  kPyNativeOutput,
  kWorkSpace,
  kOther
};

class FXRT_EXPORT Lock {
 public:
  inline void lock() {
    while (locked.test_and_set(std::memory_order_acquire)) {
    }
  }
  inline void unlock() {
    locked.clear(std::memory_order_release);
  }

 protected:
  std::atomic_flag locked = ATOMIC_FLAG_INIT;
};

class FXRT_EXPORT LockGuard {
 public:
  explicit LockGuard(const Lock& lock) : lock_(const_cast<Lock*>(&lock)) {
    lock_->lock();
  }
  ~LockGuard() {
    lock_->unlock();
  }

 private:
  Lock* lock_;
};

FXRT_EXPORT std::string MemTypeToStr(MemType memType);

constexpr size_t kPoolGrowSize = 1 << 20;

template <class T>
class ObjectPool {
  struct Buf {
    Buf* next_;
  };

  class Buffer {
    static const std::size_t bucketSize = sizeof(T) > sizeof(Buf) ? sizeof(T) : sizeof(Buf);
    static const std::size_t kDataBucketSize = bucketSize * kPoolGrowSize;

   public:
    explicit Buffer(Buffer* next) : next_(next) {}

    T* GetBlock(std::size_t index) {
      if (index >= kPoolGrowSize) {
        throw std::bad_alloc();
      }
      return reinterpret_cast<T*>(&data_[bucketSize * index]);
    }

    Buffer* const next_;

   private:
    uint8_t data_[kDataBucketSize];
  };

  Buf* freeList_ = nullptr;
  Buffer* bufferHead_ = nullptr;
  std::size_t bufferIndex_ = kPoolGrowSize;

 public:
  ObjectPool() = default;
  ObjectPool(ObjectPool&& objectPool) = delete;
  ObjectPool(const ObjectPool& objectPool) = delete;
  ObjectPool operator=(const ObjectPool& objectPool) = delete;
  ObjectPool operator=(ObjectPool&& objectPool) = delete;

  ~ObjectPool() {
    while (bufferHead_ != nullptr) {
      Buffer* buffer = bufferHead_;
      bufferHead_ = buffer->next_;
      delete buffer;
    }
  }

  T* Borrow() {
    if (freeList_ != nullptr) {
      Buf* buf = freeList_;
      freeList_ = buf->next_;
      return reinterpret_cast<T*>(buf);
    }

    if (bufferIndex_ >= kPoolGrowSize) {
      bufferHead_ = new Buffer(bufferHead_);
      bufferIndex_ = 0;
    }

    return bufferHead_->GetBlock(bufferIndex_++);
  }

  void Return(T* obj) {
    Buf* buf = reinterpret_cast<Buf*>(obj);
    buf->next_ = freeList_;
    freeList_ = buf;
  }
};

// Not support older windows version.
template <class T>
class PooledAllocator : private ObjectPool<T> {
 public:
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;

  template <class U>
  struct rebind {
    typedef PooledAllocator<U> other;
  };

  pointer allocate(size_type n, const void* hint = 0) {
    if (n != 1 || hint)
      throw std::bad_alloc();
    return ObjectPool<T>::Borrow();
  }

  void deallocate(pointer p, size_type n) {
    ObjectPool<T>::Return(p);
  }

  void construct(pointer p, const_reference val) {
    new (p) T(val);
  }

  void destroy(pointer p) {
    p->~T();
  }
};
} // namespace mem_pool
} // namespace memory
} // namespace fxrt
#endif
