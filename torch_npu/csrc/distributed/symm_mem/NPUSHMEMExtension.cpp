#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/logging/LogContext.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSymmetricMemoryUtils.hpp"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMInterface.h"
#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMExtension.h"

namespace c10d::npushmem_extension {

static std::shared_ptr<npu_logging::Logger> logger = npu_logging::logging().getLogger("torch_npu.symmetric_memory");
using c10d::symmetric_memory::NPUStoreExchange;
static NPUStoreExchange storeExchange = NPUStoreExchange("npushmem_ext");

void initialize_npushmem_with_store(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int world_size)
{
    static bool is_initialized = false;
    if (is_initialized) {
        return;
    }

    logger->debug("NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, rank is %d, world_size is %d.", rank, world_size);

    uint32_t status = c10d::symmetric_memory::Aclshmemx_set_conf_store_tls(false, nullptr, 0);
    TORCH_CHECK(status == 0, "shmem_set_conf_store_tls failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));

    int64_t init_size = c10_npu::option::OptionsManager::GetShmemSymmetricSize();
    if (c10d::symmetric_memory::Aclshmemx_get_uniqueid_exist()) {
        // gitcode version
        aclshmemx_uniqueid_t unique_id;
        if (rank == 0) {
            status = c10d::symmetric_memory::Aclshmemx_get_uniqueid(&unique_id);
            TORCH_CHECK(status == 0, "aclshmemx_get_uniqueid failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));
            logger->debug("NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, aclshmemx_get_uniqueid rank is %d, version %d, internal is %s.",
                rank, unique_id.version, unique_id.internal);
        }
        auto unique_ids = storeExchange.all_gather(store, rank, world_size, unique_id);
        logger->debug("NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, unique_id rank is %d, version %d, internal is %s.",
            rank, unique_ids[0].version, unique_ids[0].internal);

        aclshmemx_init_attr_t attr;
        logger->debug("NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, start aclshmemx_set_attr_uniqueid_args rank is %d, world_size is %d, size is %llu.",
            rank, world_size, init_size);

        status = c10d::symmetric_memory::Aclshmemx_set_attr_uniqueid_args(rank, world_size, init_size, &unique_ids[0], &attr);
        TORCH_CHECK(status == 0, "aclshmemx_set_attr_uniqueid_args failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));

        status = c10d::symmetric_memory::Aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
        TORCH_CHECK(status == 0, "aclshmemx_init_attr failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));
        logger->debug("NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store success, rank is %d, world_size is %d.", rank, world_size);
    } else {
        shmem_uniqueid_t unique_id;
        if (rank == 0) {
            status = c10d::symmetric_memory::Shmemx_get_uniqueid(&unique_id);
            TORCH_CHECK(status == 0, "shmem_get_uniqueid failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));
            logger->debug("NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, Shmem_get_uniqueid rank is %d, version %d, internal is %s.",
                rank, unique_id.version, unique_id.internal);
        }
        auto unique_ids = storeExchange.all_gather(store, rank, world_size, unique_id);
        logger->debug("NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, unique_id rank is %d, version %d, internal is %s.",
            rank, unique_ids[0].version, unique_ids[0].internal);

        shmem_init_attr_t* attributes;
        logger->debug("NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, start shmem_set_attr rank is %d, world_size is %d, size is %llu.",
            rank, world_size, init_size);
        status = c10d::symmetric_memory::Shmem_set_attr(rank, world_size, init_size, nullptr, &attributes);
        TORCH_CHECK(status == 0, "shmem_set_attr failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));
        logger->debug("NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store, end shmem_set_attr rank is %d, world_size is %d, size is %llu.",
            rank, world_size, init_size);

        status = c10d::symmetric_memory::Shmem_set_attr_uniqueid_args(rank, world_size, &unique_ids[0], attributes);
        TORCH_CHECK(status == 0, "shmem_set_attr_uniqueid_args failed, status is ", status, DIST_ERROR(ErrCode::INTERNAL));
        logger->debug("NPUSHMEMSymmetricMemoryAllocator initialize_npushmem_with_store success, rank is %d, world_size is %d.", rank, world_size);
    }
    is_initialized = true;
}

void nvshmem_put(at::Tensor& tensor, int64_t peer)
{
    // to be done: support non-contiguous tensors
    TORCH_CHECK(tensor.is_contiguous(),
        "put op currently supports contiguous tensors only", DIST_ERROR(ErrCode::PARAM));
    // to be done: rendezvous should remember the group name
    auto hdl = c10d::symmetric_memory::rendezvous(tensor, "0");
    auto rank = hdl->get_rank();
    void* buffer_ptr = hdl->get_buffer_ptrs()[rank];
    auto buffer_size = tensor.numel() * tensor.element_size();

    at::DeviceGuard device_guard(tensor.device());
    // to be done for putmem
    throw std::runtime_error("NPUSHMEMSymmetricMemory does not support nvshmem_put" + DIST_ERROR(ErrCode::NOT_SUPPORT));
}

} // namespace c10d::npushmem_extension


TORCH_LIBRARY_IMPL(symm_mem, PrivateUse1, m) {
    m.impl("nvshmem_put", c10d::npushmem_extension::nvshmem_put);
}
