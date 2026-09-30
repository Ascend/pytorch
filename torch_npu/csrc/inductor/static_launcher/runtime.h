#pragma once

#ifndef BUILD_LIBTORCH

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <acl/acl_rt.h>
#include <torch/csrc/utils/pybind.h>

namespace torch_npu::inductor {

namespace py = pybind11;

enum class StaticNpuArgKind {
  Tensor,
  I8,
  I16,
  I32,
  I64,
  U8,
  U16,
  U32,
  U64,
  F32,
  F64,
  Bool,
};

struct StaticNpuArgLayout {
  StaticNpuArgKind kind;
  size_t offset;
};

class StaticNpuKernel : public std::enable_shared_from_this<StaticNpuKernel> {
 public:
  static bool IsSupported();

  static std::shared_ptr<StaticNpuKernel> Load(
      const py::bytes& binary,
      const std::string& kernelName,
      int device,
      const std::vector<std::string>& argKinds,
      const std::string& mixMode,
      bool enableSimt,
      uint64_t sharedMemDynamicSize,
      bool isPureSimt,
      bool targetSupportFfts,
      uint64_t trailingPointerCount);

  ~StaticNpuKernel();

  void Launch(uint32_t grid0, uint32_t grid1, uint32_t grid2, uint64_t stream, const py::sequence& args);
  void Close();

 private:
  StaticNpuKernel() = default;

  void BuildPackedLayout(const std::vector<std::string>& argKinds);

  std::string kernelName_;
  int device_ = 0;
  aclrtContext context_ = nullptr;
  aclrtBinHandle binaryHandle_ = nullptr;
  aclrtFuncHandle functionHandle_ = nullptr;
  std::vector<StaticNpuArgLayout> argLayouts_;
  size_t fftsOffset_ = 0;
  size_t gridOffsets_[3] = {0, 0, 0};
  size_t packedArgsSize_ = 0;
  bool enableSimt_ = false;
  uint32_t sharedMemDynamicSize_ = 0;
  bool isPureSimt_ = false;
  bool targetSupportFfts_ = false;
  size_t trailingPointerCount_ = 0;
  void* fftsAddress_ = nullptr;
  std::mutex mutex_;
  bool closed_ = false;
};

} // namespace torch_npu::inductor

#endif
