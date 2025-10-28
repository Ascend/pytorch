#pragma once

#include <cstddef>
#include <cstdint>
#include "third_party/shmem/include/shmem_host_def.h"

namespace c10d {
namespace symmetric_memory {

int32_t Shmem_set_conf_store_tls(bool enable, const char *tls_info, const uint32_t tls_info_len);

int32_t Shmem_set_attr(int32_t my_rank, int32_t n_ranks, uint64_t local_mem_size, const char *ip_port,
                       shmem_init_attr_t **attributes);

int32_t Shmem_init_attr(shmem_init_attr_t *attributes);

int Shmem_get_uniqueid(shmem_uniqueid_t *uid);

int Shmem_set_attr_uniqueid_args(int rank_id, int nranks, const shmem_uniqueid_t *uid, shmem_init_attr_t *attr);

void *Shmem_malloc(size_t size);

void Shmem_free(void *ptr);

void *Shmem_ptr(void *ptr, int pe);

bool Shmem_finalize_exist();

int Shmem_finalize(void);

} // namespace symmetric_memory
} // namespace c10d
