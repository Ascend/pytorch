#ifndef __RUNTIME_EXCEPTION_H__
#define __RUNTIME_EXCEPTION_H__

#include <exception>
#include <mutex>
#include "common/common.h"

namespace fxrt {
namespace runtime {
class FxrtException {
 public:
  static FxrtException& GetInstance();
  void SetException(const std::exception_ptr& exception = nullptr);
  void CheckException();

 private:
  FxrtException() = default;
  ~FxrtException() = default;
  DISABLE_COPY_AND_ASSIGN(FxrtException)
  std::mutex mtx_;
  std::exception_ptr exception_{nullptr};
};
} // namespace runtime
} // namespace fxrt
#endif // __RUNTIME_EXCEPTION_H__
