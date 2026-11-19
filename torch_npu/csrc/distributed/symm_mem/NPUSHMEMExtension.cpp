#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/logging/LogContext.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSymmetricMemoryUtils.hpp"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMInterface.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMExtension.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMSymmetricMemory.hpp"

namespace c10d::npushmem_extension {

using c10d::symmetric_memory::NPUStoreExchange;
static NPUStoreExchange storeExchange = NPUStoreExchange("npushmem_ext");

void initialize_npushmem_with_store(c10::intrusive_ptr<c10d::Store> store, int rank, int world_size) {
  static bool is_initialized = false;
  if (is_initialized) {
    return;
  }

  TORCH_NPU_SYMMEM_LOGD(
      "NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, rank is %d, world_size is %d.",
      rank,
      world_size);

  uint32_t status = c10d::symmetric_memory::Aclshmemx_set_conf_store_tls(false, nullptr, 0);
  TORCH_CHECK(status == 0, "shmem_set_conf_store_tls failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));

  int64_t init_size = c10_npu::option::OptionsManager::GetShmemSymmetricSize();
  if (c10d::symmetric_memory::Aclshmemx_get_uniqueid_exist()) {
    // gitcode version
    aclshmemx_uniqueid_t unique_id;
    if (rank == 0) {
      status = c10d::symmetric_memory::Aclshmemx_get_uniqueid(&unique_id);
      TORCH_CHECK(status == 0, "aclshmemx_get_uniqueid failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));
      TORCH_NPU_SYMMEM_LOGD(
          "NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, aclshmemx_get_uniqueid rank is %d, version %d, internal is %s.",
          rank,
          unique_id.version,
          unique_id.internal);
    }
    auto unique_ids = storeExchange.all_gather(store, rank, world_size, unique_id);
    TORCH_NPU_SYMMEM_LOGD(
        "NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, unique_id rank is %d, version %d, internal is %s.",
        rank,
        unique_ids[0].version,
        unique_ids[0].internal);

    aclshmemx_init_attr_t attr;
    TORCH_NPU_SYMMEM_LOGD(
        "NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, start aclshmemx_set_attr_uniqueid_args rank is %d, world_size is %d, size is %llu.",
        rank,
        world_size,
        init_size);

    status =
        c10d::symmetric_memory::Aclshmemx_set_attr_uniqueid_args(rank, world_size, init_size, &unique_ids[0], &attr);
    TORCH_CHECK(
        status == 0, "aclshmemx_set_attr_uniqueid_args failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));

    status = c10d::symmetric_memory::Aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);
    TORCH_CHECK(status == 0, "aclshmemx_init_attr failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));
    TORCH_NPU_SYMMEM_LOGD(
        "NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store success, rank is %d, world_size is %d.",
        rank,
        world_size);
  } else {
    shmem_uniqueid_t unique_id;
    if (rank == 0) {
      status = c10d::symmetric_memory::Shmemx_get_uniqueid(&unique_id);
      TORCH_CHECK(status == 0, "shmem_get_uniqueid failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));
      TORCH_NPU_SYMMEM_LOGD(
          "NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, Shmem_get_uniqueid rank is %d, version %d, internal is %s.",
          rank,
          unique_id.version,
          unique_id.internal);
    }
    auto unique_ids = storeExchange.all_gather(store, rank, world_size, unique_id);
    TORCH_NPU_SYMMEM_LOGD(
        "NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, unique_id rank is %d, version %d, internal is %s.",
        rank,
        unique_ids[0].version,
        unique_ids[0].internal);

    shmem_init_attr_t* attributes;
    TORCH_NPU_SYMMEM_LOGD(
        "NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, start shmem_set_attr rank is %d, world_size is %d, size is %llu.",
        rank,
        world_size,
        init_size);
    status = c10d::symmetric_memory::Shmem_set_attr(rank, world_size, init_size, nullptr, &attributes);
    TORCH_CHECK(status == 0, "shmem_set_attr failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));
    TORCH_NPU_SYMMEM_LOGD(
        "NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, end shmem_set_attr rank is %d, world_size is %d, size is %llu.",
        rank,
        world_size,
        init_size);

    status = c10d::symmetric_memory::Shmem_set_attr_uniqueid_args(rank, world_size, &unique_ids[0], attributes);
    TORCH_CHECK(status == 0, "shmem_set_attr_uniqueid_args failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));
    TORCH_NPU_SYMMEM_LOGD(
        "NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store success, rank is %d, world_size is %d.",
        rank,
        world_size);
  }
  is_initialized = true;
}

void npushmem_put(at::Tensor& tensor, int64_t peer) {
  // to be done: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(), "put op currently supports contiguous tensors only", DIST_ERROR(ErrCode::PARAM));
  // to be done: rendezvous should remember the group name
  auto hdl = c10d::symmetric_memory::rendezvous(tensor, "0");
  auto rank = hdl->get_rank();
  void* buffer_ptr = hdl->get_buffer_ptrs()[rank];
  auto buffer_size = tensor.numel() * tensor.element_size();
  TORCH_CHECK(
      peer >= 0 && peer < hdl->get_world_size(), "peer must be in range [0, world size)", DIST_ERROR(ErrCode::PARAM));
  at::DeviceGuard device_guard(tensor.device());
  auto stream = c10_npu::getCurrentNPUStream();
  c10d::symmetric_memory::Shmem_putmem_on_stream(buffer_ptr, tensor.data_ptr(), buffer_size, peer, stream);
}

void npushmem_get(at::Tensor& tensor, int64_t peer) {
  // to be done: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(), "get op currently supports contiguous tensors only", DIST_ERROR(ErrCode::PARAM));
  // to be done: rendezvous should remember the group name
  auto hdl = c10d::symmetric_memory::rendezvous(tensor, "0");
  auto rank = hdl->get_rank();
  void* buffer_ptr = hdl->get_buffer_ptrs()[rank];
  auto buffer_size = tensor.numel() * tensor.element_size();
  TORCH_CHECK(
      peer >= 0 && peer < hdl->get_world_size(), "peer must be in range [0, world size)", DIST_ERROR(ErrCode::PARAM));
  at::DeviceGuard device_guard(tensor.device());
  auto stream = c10_npu::getCurrentNPUStream();
  c10d::symmetric_memory::Shmem_getmem_on_stream(tensor.mutable_data_ptr(), buffer_ptr, buffer_size, peer, stream);
}

// Non-blocking put: submitted on the libshmem-managed default stream, so the
// caller's stream is drained first; completion is guaranteed by npushmem_quiet.
void npushmem_put_nbi(at::Tensor& tensor, int64_t peer) {
  // to be done: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(), "put op currently supports contiguous tensors only", DIST_ERROR(ErrCode::PARAM));
  auto hdl = c10d::symmetric_memory::rendezvous(tensor, "0");
  auto rank = hdl->get_rank();
  void* buffer_ptr = hdl->get_buffer_ptrs()[rank];
  auto buffer_size = tensor.numel() * tensor.element_size();
  TORCH_CHECK(
      peer >= 0 && peer < hdl->get_world_size(), "peer must be in range [0, world size)", DIST_ERROR(ErrCode::PARAM));
  at::DeviceGuard device_guard(tensor.device());
  c10_npu::getCurrentNPUStream().synchronize();
  c10d::symmetric_memory::Aclshmem_putmem_nbi(buffer_ptr, tensor.data_ptr(), buffer_size, peer);
}

// Non-blocking get with the same ordering and quiet-completion contract as npushmem_put_nbi.
void npushmem_get_nbi(at::Tensor& tensor, int64_t peer) {
  // to be done: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(), "get op currently supports contiguous tensors only", DIST_ERROR(ErrCode::PARAM));
  auto hdl = c10d::symmetric_memory::rendezvous(tensor, "0");
  auto rank = hdl->get_rank();
  void* buffer_ptr = hdl->get_buffer_ptrs()[rank];
  auto buffer_size = tensor.numel() * tensor.element_size();
  TORCH_CHECK(
      peer >= 0 && peer < hdl->get_world_size(), "peer must be in range [0, world size)", DIST_ERROR(ErrCode::PARAM));
  at::DeviceGuard device_guard(tensor.device());
  c10_npu::getCurrentNPUStream().synchronize();
  c10d::symmetric_memory::Aclshmem_getmem_nbi(tensor.mutable_data_ptr(), buffer_ptr, buffer_size, peer);
}

// Blocks until all nbi operations complete; the tensor is only a dispatch
// anchor. Falls back to quiet_on_stream(nullptr) + device sync without aclshmem_quiet.
void npushmem_quiet(const at::Tensor& tensor) {
  at::DeviceGuard device_guard(tensor.device());
  if (c10d::symmetric_memory::Aclshmem_quiet_exist()) {
    c10d::symmetric_memory::Aclshmem_quiet();
    return;
  }
  c10d::symmetric_memory::Aclshmemx_quiet_on_stream(nullptr);
  c10_npu::npuSynchronizeDevice();
}

} // namespace c10d::npushmem_extension

// Upstream owns the legacy nvshmem_* schemas and defines no npushmem_* names:
// define the five npushmem_* schemas here, above the impl block (init order).
TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def("npushmem_put(Tensor(a!) tensor, int peer) -> ()");
  m.def("npushmem_get(Tensor(a!) tensor, int peer) -> ()");
  m.def("npushmem_put_nbi(Tensor(a!) tensor, int peer) -> ()");
  m.def("npushmem_get_nbi(Tensor(a!) tensor, int peer) -> ()");
  m.def("npushmem_quiet(Tensor tensor) -> ()");
}

namespace c10d::npushmem_extension {

// Deprecated nvshmem_* aliases: warn once per process, then forward to the
// npushmem_* implementation.
void nvshmem_put(at::Tensor& tensor, int64_t peer) {
  TORCH_WARN_ONCE(
      "torch.ops.symm_mem.nvshmem_put is deprecated, please use "
      "torch.ops.symm_mem.npushmem_put instead.");
  npushmem_put(tensor, peer);
}

void nvshmem_get(at::Tensor& tensor, int64_t peer) {
  TORCH_WARN_ONCE(
      "torch.ops.symm_mem.nvshmem_get is deprecated, please use "
      "torch.ops.symm_mem.npushmem_get instead.");
  npushmem_get(tensor, peer);
}

} // namespace c10d::npushmem_extension

TORCH_LIBRARY_IMPL(symm_mem, PrivateUse1, m) {
  m.impl("npushmem_put", c10d::npushmem_extension::npushmem_put);
  m.impl("npushmem_get", c10d::npushmem_extension::npushmem_get);
  m.impl("npushmem_put_nbi", c10d::npushmem_extension::npushmem_put_nbi);
  m.impl("npushmem_get_nbi", c10d::npushmem_extension::npushmem_get_nbi);
  m.impl("npushmem_quiet", c10d::npushmem_extension::npushmem_quiet);
  // Deprecated aliases for the upstream-owned nvshmem_* schemas.
  m.impl("nvshmem_put", c10d::npushmem_extension::nvshmem_put);
  m.impl("nvshmem_get", c10d::npushmem_extension::nvshmem_get);
}
