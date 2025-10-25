#include "torch_npu/csrc/distributed/symm_mem/NPUSHMEMInterface.h"
#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace c10d {
namespace symmetric_memory {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libshmem, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)           \
  GET_FUNCTION(libshmem, funcName)

REGISTER_LIBRARY(libshmem)
LOAD_FUNCTION(shmem_set_conf_store_tls)
LOAD_FUNCTION(shmem_set_attr)
LOAD_FUNCTION(shmem_init_attr)
LOAD_FUNCTION(shmem_get_uniqueid)
LOAD_FUNCTION(shmem_set_attr_uniqueid_args)
LOAD_FUNCTION(shmem_malloc)
LOAD_FUNCTION(shmem_free)
LOAD_FUNCTION(shmem_ptr)
LOAD_FUNCTION(shmem_finalize)

int32_t Shmem_set_conf_store_tls(bool enable, const char *tls_info, const uint32_t tls_info_len)
{
    typedef int32_t (*ShmemApiFunc)(bool, const char *, const uint32_t);
    static ShmemApiFunc shmem_set_conf_store_tls_func = nullptr;
    if (shmem_set_conf_store_tls_func == nullptr) {
        shmem_set_conf_store_tls_func = (ShmemApiFunc)GET_FUNC(shmem_set_conf_store_tls);
    }
    TORCH_CHECK(shmem_set_conf_store_tls_func, "Failed to find function ", "shmem_set_conf_store_tls", PTA_ERROR(ErrCode::NOT_FOUND));
    return shmem_set_conf_store_tls_func(enable, tls_info, tls_info_len);
}

int32_t Shmem_set_attr(int32_t my_rank, int32_t n_ranks, uint64_t local_mem_size, const char *ip_port,
                       shmem_init_attr_t **attributes)
{
    typedef int32_t (*ShmemApiFunc)(int32_t, int32_t, uint64_t, const char *, shmem_init_attr_t **);
    static ShmemApiFunc shmem_set_attr_func = nullptr;
    if (shmem_set_attr_func == nullptr) {
        shmem_set_attr_func = (ShmemApiFunc)GET_FUNC(shmem_set_attr);
    }
    TORCH_CHECK(shmem_set_attr_func, "Failed to find function ", "shmem_set_attr", PTA_ERROR(ErrCode::NOT_FOUND));
    return shmem_set_attr_func(my_rank, n_ranks, local_mem_size, ip_port, attributes);
}

int32_t Shmem_init_attr(shmem_init_attr_t *attributes)
{
    typedef int32_t (*ShmemApiFunc)(shmem_init_attr_t *);
    static ShmemApiFunc shmem_init_attr_func = nullptr;
    if (shmem_init_attr_func == nullptr) {
        shmem_init_attr_func = (ShmemApiFunc)GET_FUNC(shmem_init_attr);
    }
    TORCH_CHECK(shmem_init_attr_func, "Failed to find function ", "shmem_init_attr", PTA_ERROR(ErrCode::NOT_FOUND));
    return shmem_init_attr_func(attributes);
}

int32_t Shmem_get_uniqueid(shmem_uniqueid_t *uid)
{
    typedef int32_t (*ShmemApiFunc)(shmem_uniqueid_t *);
    static ShmemApiFunc shmem_get_uniqueid_func = nullptr;
    if (shmem_get_uniqueid_func == nullptr) {
        shmem_get_uniqueid_func = (ShmemApiFunc)GET_FUNC(shmem_get_uniqueid);
    }
    TORCH_CHECK(shmem_get_uniqueid_func, "Failed to find function ", "shmem_get_uniqueid", PTA_ERROR(ErrCode::NOT_FOUND));
    return shmem_get_uniqueid_func(uid);
}

int Shmem_set_attr_uniqueid_args(int rank_id, int nranks, const shmem_uniqueid_t *uid, shmem_init_attr_t *attr)
{
    typedef int32_t (*ShmemApiFunc)(int, int, const shmem_uniqueid_t *, shmem_init_attr_t *);
    static ShmemApiFunc shmem_set_attr_uniqueid_args_func = nullptr;
    if (shmem_set_attr_uniqueid_args_func == nullptr) {
        shmem_set_attr_uniqueid_args_func = (ShmemApiFunc)GET_FUNC(shmem_set_attr_uniqueid_args);
    }
    TORCH_CHECK(shmem_set_attr_uniqueid_args_func, "Failed to find function ", "shmem_set_attr_uniqueid_args", PTA_ERROR(ErrCode::NOT_FOUND));
    return shmem_set_attr_uniqueid_args_func(rank_id, nranks, uid, attr);
}

void *Shmem_malloc(size_t size)
{
    typedef void* (*ShmemApiFunc)(size_t);
    static ShmemApiFunc shmem_malloc_func = nullptr;
    if (shmem_malloc_func == nullptr) {
        shmem_malloc_func = (ShmemApiFunc)GET_FUNC(shmem_malloc);
    }
    TORCH_CHECK(shmem_malloc_func, "Failed to find function ", "shmem_malloc", PTA_ERROR(ErrCode::NOT_FOUND));
    return shmem_malloc_func(size);
}

void Shmem_free(void *ptr)
{
    typedef void (*ShmemApiFunc)(void *);
    static ShmemApiFunc shmem_free_func = nullptr;
    if (shmem_free_func == nullptr) {
        shmem_free_func = (ShmemApiFunc)GET_FUNC(shmem_free);
    }
    TORCH_CHECK(shmem_free_func, "Failed to find function ", "shmem_free", PTA_ERROR(ErrCode::NOT_FOUND));
    return shmem_free_func(ptr);
}

void *Shmem_ptr(void *ptr, int pe)
{
    typedef void* (*ShmemApiFunc)(void *, int);
    static ShmemApiFunc shmem_ptr_func = nullptr;
    if (shmem_ptr_func == nullptr) {
        shmem_ptr_func = (ShmemApiFunc)GET_FUNC(shmem_ptr);
    }
    TORCH_CHECK(shmem_ptr_func, "Failed to find function ", "shmem_ptr", PTA_ERROR(ErrCode::NOT_FOUND));
    return shmem_ptr_func(ptr, pe);
}

bool Shmem_finalize_exist()
{
    const static bool shmemApiFuncExist = []() -> bool {
        try {
            auto func = GET_FUNC(shmem_finalize)
            return func != nullptr;
        } catch (...) {
            // libshmem.so not exist
            return false;
        }
    }();
    return shmemApiFuncExist;
}

int Shmem_finalize(void)
{
    typedef int (*ShmemApiFunc)(void);
    static ShmemApiFunc shmem_finalize_func = nullptr;
    if (shmem_finalize_func == nullptr) {
        shmem_finalize_func = (ShmemApiFunc)GET_FUNC(shmem_finalize);
    }
    TORCH_CHECK(shmem_finalize_func, "Failed to find function ", "shmem_finalize", PTA_ERROR(ErrCode::NOT_FOUND));
    return shmem_finalize_func();
}

} // namespace symmetric_memory
} // namespace c10d
