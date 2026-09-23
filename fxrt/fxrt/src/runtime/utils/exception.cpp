#include "runtime/utils/exception.h"
#include "common/logger.h"

namespace fxrt {
namespace runtime {
FxrtException& FxrtException::GetInstance() {
  static FxrtException instance{};
  return instance;
}

void FxrtException::SetException(const std::exception_ptr& exception) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (exception_ != nullptr) {
    return;
  }

  if (exception != nullptr) {
    exception_ = exception;
  } else {
    exception_ = std::current_exception();
  }
}

void FxrtException::CheckException() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (exception_ != nullptr) {
    auto exception = exception_;
    exception_ = nullptr;
    std::rethrow_exception(exception);
  }
}
} // namespace runtime
} // namespace fxrt
