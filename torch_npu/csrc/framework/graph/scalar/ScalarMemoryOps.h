#pragma once

#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

#define HOST_MEM_INIT_SIZE (512 * 10240)    // 5M
#define CHECK_MEM_MAX_SIZE (65536 * 1024)   // 64M
#define DEVICE_VALID_LEN(a) ((((a) + 32 + 511) / 512) * 512)

class  ScalarMemContext {
public:
  static ScalarMemContext &GetContext() {
    static ScalarMemContext ctx;
    return ctx;
  }

  ScalarMemContext(const ScalarMemContext&) = delete;
  ScalarMemContext(ScalarMemContext&&) = delete;
  ScalarMemContext& operator=(const ScalarMemContext&) = delete;
  ScalarMemContext& operator=(ScalarMemContext&&) = delete;

  uint8_t* GetDeviceMemBuffer() {
    return reinterpret_cast<uint8_t*>(npu_tensor_.data_ptr());
  }

  void AppendToHostMem(
      uint8_t* host_ptr,
      uint32_t data_len,
      uint32_t& data_offset);

  void ExecuteH2D(c10_npu::NPUStream stream);

  void Reset();

private:
  void Init();

  void CheckForExpand(uint32_t input_valid_len);

  ScalarMemContext() = default;

  bool inited_ = false;
  at::Tensor cpu_tensor_;
  at::Tensor npu_tensor_;
  uint32_t host_mem_valid_len_ = 0;
};

} // namespace native
} // namespace at_npu