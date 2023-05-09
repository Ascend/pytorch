#include "hccl.h"

hcclResult_t HcclCommInitUniqueId(hcclComm_t* comm, u32 nranks, hcclUniqueId commId, u32 myrank) {return HCCL_SUCCESS;}
hcclResult_t HcclGetUniqueId(hcclUniqueId* id) {return HCCL_SUCCESS;}
hcclResult_t HcclAllReduce(void *inputPtr, void *outputPtr, u64 count, hcclDataType_t dataType,
                                  hcclRedOp_t op, hcclComm_t comm, rtStream_t stream) {return HCCL_SUCCESS;}
hcclResult_t HcclBroadcast(void *ptr, u64 count, hcclDataType_t dataType, u32 root, hcclComm_t comm,
                                  rtStream_t stream) {return HCCL_SUCCESS;}
hcclResult_t HcclCommDestroy(hcclComm_t comm) {return HCCL_SUCCESS;}