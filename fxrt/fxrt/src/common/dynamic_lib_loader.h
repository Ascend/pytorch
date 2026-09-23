#ifndef __COMMON_DYNAMIC_LIB_LOADER_H__
#define __COMMON_DYNAMIC_LIB_LOADER_H__

#include <string>
#include <map>
#include <utility>
#include "common/common.h"
#include "common/visible.h"

namespace fxrt {
namespace common {
class DA_API DynamicLibLoader {
 public:
  DynamicLibLoader() {
    filePath_ = GetFilePathFromDlInfo();
    if (filePath_.empty()) {
      RT_GLOG(ERROR) << "Get dynamic library file path by dladdr failed";
    }
  }
  explicit DynamicLibLoader(const std::string&& filePath) : filePath_(std::move(filePath)) {}
  ~DynamicLibLoader();

  bool LoadDynamicLib(const std::string& dlName, std::stringstream* errMsg);
  void CloseDynamicLib(const std::string& dlName);
  void* GetHandle(const std::string& dlName) const;

  const std::string& GetDynamicLibFilePath() const {
    return filePath_;
  }

 private:
  DISABLE_COPY_AND_ASSIGN(DynamicLibLoader)
  static std::string GetFilePathFromDlInfo();
  std::map<std::string, void*> allHandles_;
  std::string filePath_;
};
} // namespace common
} // namespace fxrt

#endif // __COMMON_DYNAMIC_LIB_LOADER_H__
