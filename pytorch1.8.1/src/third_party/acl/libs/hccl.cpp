#include "hccl.h"

hcclResult_t hcclCommInitUniqueId(hcclComm_t* comm, u32 nranks, hcclUniqueId commId, u32 myrank) {return HCCL_SUCCESS;}
hcclResult_t hcclGetUniqueId(hcclUniqueId* id) {return HCCL_SUCCESS;}
hcclResult_t hcclAllReduce(void *inputPtr, void *outputPtr, u64 count, hcclDataType_t dataType,
                                  hcclRedOp_t op, hcclComm_t comm, rtStream_t stream) {return HCCL_SUCCESS;}
hcclResult_t hcclBroadcast(void *ptr, u64 count, hcclDataType_t dataType, u32 root, hcclComm_t comm,
                                  rtStream_t stream) {return HCCL_SUCCESS;}
hcclResult_t hcclCommDestroy(hcclComm_t comm) {return HCCL_SUCCESS;}