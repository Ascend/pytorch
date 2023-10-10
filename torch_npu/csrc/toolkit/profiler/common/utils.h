#ifndef TORCH_NPU_TOOLKIT_PROFILER_UTILS_INC
#define TORCH_NPU_TOOLKIT_PROFILER_UTILS_INC
#include <unistd.h>
#include <sys/stat.h>
#include <linux/limits.h>
#include <libgen.h>
#include <fcntl.h>

#include <stdint.h>

#include <string>

namespace torch_npu {
namespace toolkit {
namespace profiler {
class Utils {
public:
  static bool IsFileExist(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    return (access(path.c_str(), F_OK) == 0) ? true : false;
  }

  static bool IsFileWritable(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    return (access(path.c_str(), W_OK) == 0) ? true : false;
  }

  static bool IsDir(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    struct stat st = {0};
    int ret = lstat(path.c_str(), &st);
    if (ret != 0) {
      return false;
    }
    return S_ISDIR(st.st_mode) ? true : false;
  }

  static bool CreateDir(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return false;
    }
    if (IsFileExist(path)) {
      return IsDir(path) ? true : false;
    }
    size_t pos = 0;
    while ((pos = path.find_first_of('/', pos)) != std::string::npos) {
      std::string base_dir = path.substr(0, ++pos);
      if (IsFileExist(base_dir)) {
        if (IsDir(base_dir)) {
          continue;
        } else {
          return false;
        }
      }
      if (mkdir(base_dir.c_str(), 0750) != 0) {
        return false;
      }
    }
    return (mkdir(path.c_str(), 0750) == 0) ? true : false;
  }

  static std::string RealPath(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return "";
    }
    char realPath[PATH_MAX] = {0};
    if (realpath(path.c_str(), realPath) == nullptr) {
      return "";
    }
    return std::string(realPath);
  }

  static std::string RelativeToAbsPath(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX) {
      return "";
    }
    if (path[0] != '/') {
      char pwd_path[PATH_MAX] = {0};
      if (getcwd(pwd_path, PATH_MAX) != nullptr) {
        return std::string(pwd_path) + "/" + path;
      }
      return "";
    }
    return std::string(path);
  }

  static std::string DirName(const std::string &path) {
    if (path.empty()) {
      return "";
    }
    std::string temp_path = path;
    char *path_c = dirname(const_cast<char *>(temp_path.data()));
    return path_c ? std::string(path_c) : "";
  }

  static int64_t GetClockMonotonicRawNs() {
    struct timespec ts = {0};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000 + static_cast<int64_t>(ts.tv_nsec);
  }

  static bool CreateFile(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX || !CreateDir(DirName(path))) {
      return false;
    }
    int fd = creat(path.c_str(), S_IRUSR | S_IWUSR | S_IRGRP);
    return (fd < 0 || close(fd) != 0) ? false : true;
  }

  static bool IsSoftLink(const std::string &path) {
    if (path.empty() || path.size() > PATH_MAX || !IsFileExist(path)) {
      return false;
    }
    struct stat st{};
    if (lstat(path.c_str(), &st) != 0) {
      return false;
    }
    return S_ISLNK(st.st_mode);
  }
};
} // profiler
} // toolkit
} // torch_npu
#endif
