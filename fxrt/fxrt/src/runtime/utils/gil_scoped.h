#ifndef __RUNTIME_UTILS_GIL_SCOPED_H__
#define __RUNTIME_UTILS_GIL_SCOPED_H__

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace fxrt {
namespace runtime {
class GilReleaseWithCheck {
 public:
  GilReleaseWithCheck() {
    if (Py_IsInitialized() != 0 && PyGILState_Check() != 0) {
      release_ = std::make_unique<nb::gil_scoped_release>();
    }
  }

  ~GilReleaseWithCheck() {
    release_ = nullptr;
  }

 private:
  std::unique_ptr<nb::gil_scoped_release> release_;
};
} // namespace runtime
} // namespace fxrt

#endif // __RUNTIME_UTILS_GIL_SCOPED_H__
