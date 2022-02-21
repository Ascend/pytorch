#include "ScalarMemoryOps.h"

namespace at_npu {
namespace native {

void ScalarMemContext::Init() {
  cpu_tensor_ = at::empty(
      {HOST_MEM_INIT_SIZE},
      at::TensorOptions().pinned_memory(true).device(at::kCPU).dtype(at::kByte));
  host_mem_valid_len_ = 0;
  inited_ = true;
}

void ScalarMemContext::ExecuteH2D(c10::npu::NPUStream stream) {
  if (!inited_) {
    return;
  }

  if (CHECK_MEM_MAX_SIZE <= host_mem_valid_len_) {
    AT_ERROR("Checked the device memory size >= 64M.");
    return;
  }
  int deviceIndex = 0;
  AT_NPU_CHECK(aclrtGetDevice(&deviceIndex));
  npu_tensor_ = at::empty(
      {host_mem_valid_len_},
      at::TensorOptions().device(at::kNPU, deviceIndex).dtype(at::kByte));
  
  AT_NPU_CHECK(
      aclrtMemcpyAsync(
          npu_tensor_.data_ptr(),
          host_mem_valid_len_,
          cpu_tensor_.data_ptr(),
          host_mem_valid_len_,
          ACL_MEMCPY_HOST_TO_DEVICE,
          stream));
  AT_NPU_CHECK(THNPUCachingHostAllocator_recordEvent(cpu_tensor_.data_ptr(), stream));

  // reset pin memory
  cpu_tensor_.reset();
}

void ScalarMemContext::CheckForExpand(uint32_t input_valid_len) {
  if (input_valid_len <= (cpu_tensor_.nbytes() - host_mem_valid_len_)) {
    return;
  }

  auto tmp_tensor = cpu_tensor_;
  uint32_t expand_tensor_size = tmp_tensor.nbytes() + HOST_MEM_INIT_SIZE;
  cpu_tensor_ = at::empty(
      {expand_tensor_size},
      at::TensorOptions().pinned_memory(true).device(at::kCPU).dtype(at::kByte));
  
  AT_NPU_CHECK(
      aclrtMemcpy(
          cpu_tensor_.data_ptr(),
          host_mem_valid_len_,
          tmp_tensor.data_ptr(),
          host_mem_valid_len_,
          ACL_MEMCPY_HOST_TO_HOST));
}

void ScalarMemContext::AppendToHostMem(
    uint8_t* host_ptr,
    uint32_t data_len,
    uint32_t& data_offset) {
  if (!inited_) {
    Init();
  }

  uint32_t valid_len = DEVICE_VALID_LEN(data_len);
  CheckForExpand(valid_len);
  data_offset = host_mem_valid_len_;
  std::memcpy(
      reinterpret_cast<uint8_t*>(cpu_tensor_.data_ptr()) + data_offset,
      host_ptr, data_len);
  host_mem_valid_len_ += valid_len;
}

void ScalarMemContext::Reset() {
  npu_tensor_.reset();
  inited_ = false;
}

} // namespace native
} // namespace at_npu