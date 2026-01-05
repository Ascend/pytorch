#pragma once

#include <cstddef>
#include <cstdint>
#include "third_party/shmem/include/shmem_host_def.h"

namespace c10d {
namespace symmetric_memory {

int32_t Aclshmemx_set_conf_store_tls(bool enable, const char *tls_info, const uint32_t tls_info_len);

int32_t Shmem_set_attr(int32_t my_rank, int32_t n_ranks, uint64_t local_mem_size, const char *ip_port,
                       shmem_init_attr_t **attributes);

int Shmemx_get_uniqueid(shmem_uniqueid_t *uid);

bool Aclshmemx_get_uniqueid_exist();

int Aclshmemx_get_uniqueid(aclshmemx_uniqueid_t *uid);

int Shmem_set_attr_uniqueid_args(int rank_id, int nranks, const shmem_uniqueid_t *uid, shmem_init_attr_t *attr);

int Aclshmemx_set_attr_uniqueid_args(int rank_id, int nranks, int64_t local_mem_size, aclshmemx_uniqueid_t *uid, aclshmemx_init_attr_t *aclshmem_attr);

int Aclshmemx_init_attr(aclshmemx_bootstrap_t bootstrap_flags, aclshmemx_init_attr_t *attributes);

void *Aclshmem_malloc(size_t size);

void Aclshmem_free(void *ptr);

void *Aclshmem_ptr(void *ptr, int pe);

bool Aclshmem_finalize_exist();

int Aclshmem_finalize(void);

} // namespace symmetric_memory
} // namespace c10d
