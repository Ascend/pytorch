#ifndef __IR_COMMON_INTRUSIVE_PTR_H__
#define __IR_COMMON_INTRUSIVE_PTR_H__

#include <cstddef>
#include <functional>
#include <atomic>
#include <utility>
#include <type_traits>

namespace fxrt {
namespace ir {

/**
 * @brief Base class for reference-counted objects.
 *
 * This class provides the basic functionality for intrusive reference counting.
 * Classes that want to be managed by IntrusivePtr should inherit from this class.
 */
class RefCounted {
 public:
  /**
   * @brief Default constructor. Initializes the reference count to 0.
   */
  RefCounted() : refCount_(0) {}
  /**
   * @brief Virtual destructor.
   */
  virtual ~RefCounted() = default;

  /**
   * @brief Increments the reference count.
   */
  void AddRef() const {
    refCount_++;
  }

  /**
   * @brief Decrements the reference count. If the count reaches 0, the object is deleted.
   */
  void DecRef() const {
    if (--refCount_ == 0) {
      delete this;
    }
  }

 private:
  mutable std::atomic<size_t> refCount_;
};

/**
 * @brief A smart pointer that uses intrusive reference counting.
 *
 * IntrusivePtr is similar to std::shared_ptr, but it assumes that the managed object
 * provides its own reference counting mechanism (by inheriting from RefCounted).
 * This can be more efficient than std::shared_ptr as it avoids a separate control block.
 *
 * @tparam T The type of the managed object. Must be a subclass of RefCounted.
 */
template <typename T>
class IntrusivePtr {
 public:
  /**
   * @brief Default constructor. Creates a null IntrusivePtr.
   */
  IntrusivePtr() : ptr_(nullptr) {}

  /**
   * @brief Constructor from a raw pointer. Takes ownership of the pointer.
   * @param p The raw pointer to manage.
   */
  // NOLINTNEXTLINE(runtime/explicit)
  IntrusivePtr(T* p) : ptr_(p) {
    if (ptr_) {
      ptr_->AddRef();
    }
  }

  /**
   * @brief Copy constructor. Increments the reference count.
   * @param other The IntrusivePtr to copy from.
   */
  IntrusivePtr(const IntrusivePtr& other) : ptr_(other.ptr_) {
    if (ptr_) {
      ptr_->AddRef();
    }
  }

  /**
   * @brief Move constructor. Takes ownership from another IntrusivePtr.
   * @param other The IntrusivePtr to move from.
   */
  IntrusivePtr(IntrusivePtr&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  /**
   * @brief Conversion constructor from derived type.
   * @tparam U The derived type that inherits from T.
   * @param other The IntrusivePtr of the derived type.
   */
  template <typename U, typename = std::enable_if_t<std::is_base_of_v<T, U>>>
  IntrusivePtr(const IntrusivePtr<U>& other) : ptr_(other.get()) {
    if (ptr_) {
      ptr_->AddRef();
    }
  }

  /**
   * @brief Move conversion constructor from derived type.
   * @tparam U The derived type that inherits from T.
   * @param other The IntrusivePtr of the derived type to move from.
   */
  template <typename U, typename = std::enable_if_t<std::is_base_of_v<T, U>>>
  IntrusivePtr(IntrusivePtr<U>&& other) noexcept : ptr_(other.Release()) {}

  /**
   * @brief Destructor. Decrements the reference count.
   */
  ~IntrusivePtr() {
    if (ptr_) {
      ptr_->DecRef();
    }
  }

  /**
   * @brief Copy assignment operator.
   * @param other The IntrusivePtr to copy from.
   * @return *this
   */
  IntrusivePtr& operator=(const IntrusivePtr& other) {
    if (this != &other) {
      if (ptr_) {
        ptr_->DecRef();
      }
      ptr_ = other.ptr_;
      if (ptr_) {
        ptr_->AddRef();
      }
    }
    return *this;
  }

  /**
   * @brief Move assignment operator.
   * @param other The IntrusivePtr to move from.
   * @return *this
   */
  IntrusivePtr& operator=(IntrusivePtr&& other) noexcept {
    if (this != &other) {
      if (ptr_) {
        ptr_->DecRef();
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  /**
   * @brief Copy assignment operator from derived type.
   * @tparam U The derived type that inherits from T.
   * @param other The IntrusivePtr of the derived type to copy from.
   * @return *this
   */
  template <typename U, typename = std::enable_if_t<std::is_base_of_v<T, U>>>
  IntrusivePtr& operator=(const IntrusivePtr<U>& other) {
    if (ptr_ != other.get()) {
      if (ptr_) {
        ptr_->DecRef();
      }
      ptr_ = other.get();
      if (ptr_) {
        ptr_->AddRef();
      }
    }
    return *this;
  }

  /**
   * @brief Move assignment operator from derived type.
   * @tparam U The derived type that inherits from T.
   * @param other The IntrusivePtr of the derived type to move from.
   * @return *this
   */
  template <typename U, typename = std::enable_if_t<std::is_base_of_v<T, U>>>
  IntrusivePtr& operator=(IntrusivePtr<U>&& other) noexcept {
    if (ptr_ != other.get()) {
      if (ptr_) {
        ptr_->DecRef();
      }
      ptr_ = other.Release();
    }
    return *this;
  }

  /**
   * @brief Gets the raw pointer.
   * @return The managed pointer.
   */
  T* get() const {
    return ptr_;
  }

  /**
   * @brief Dereferences the managed pointer.
   * @return A reference to the managed object.
   */
  T& operator*() const {
    return *ptr_;
  }
  /**
   * @brief Dereferences the managed pointer for member access.
   * @return The managed pointer.
   */
  T* operator->() const {
    return ptr_;
  }

  /**
   * @brief Checks if the pointer is equal to another IntrusivePtr.
   * @param other The other IntrusivePtr to compare with.
   * @return true if the pointers are equal, false otherwise.
   */
  bool operator==(const IntrusivePtr& other) const {
    return ptr_ == other.ptr_;
  }

  /**
   * @brief Checks if the pointer is null.
   * @return true if the pointer is null, false otherwise.
   */
  bool operator==(const nullptr_t&) const {
    return ptr_ == nullptr;
  }

  /**
   * @brief Checks if the pointer is not null.
   * @return true if the pointer is not null, false otherwise.
   */
  bool operator!=(const nullptr_t&) const {
    return ptr_ != nullptr;
  }

  /**
   * @brief Checks if the pointer is not null.
   * @return true if the pointer is not null, false otherwise.
   */
  explicit operator bool() const {
    return ptr_ != nullptr;
  }

  /**
   * @brief Resets the IntrusivePtr to null, decrementing the reference count.
   */
  void Reset() {
    if (ptr_) {
      ptr_->DecRef();
      ptr_ = nullptr;
    }
  }

  /**
   * @brief Releases ownership of the managed pointer.
   * The reference count is not decremented.
   * @return The raw pointer.
   */
  T* Release() {
    T* p = ptr_;
    ptr_ = nullptr;
    return p;
  }

 private:
  T* ptr_;
};

/**
 * @brief Creates an object managed by an IntrusivePtr.
 *
 * @tparam T The type of the object to create.
 * @tparam Args The types of the arguments for the constructor of T.
 * @param args The arguments for the constructor of T.
 * @return An IntrusivePtr managing the newly created object.
 */
template <typename T, typename... Args>
IntrusivePtr<T> MakeIntrusive(Args&&... args) {
  return IntrusivePtr<T>(new T(std::forward<Args>(args)...));
}

} // namespace ir
} // namespace fxrt

// Specialization of std::hash for IntrusivePtr to allow its use in unordered containers.
template <typename T>
struct std::hash<fxrt::ir::IntrusivePtr<T>> {
  std::size_t operator()(const fxrt::ir::IntrusivePtr<T>& k) const {
    return std::hash<T*>()(k.get());
  }
};

#endif // __IR_COMMON_INTRUSIVE_PTR_H__
