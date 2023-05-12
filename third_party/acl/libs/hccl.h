extern "C" {
typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

constexpr u32 HCCL_UNIQUE_ID_BYTES =2060; // 2060: unique id length
using hcclUniqueId = struct hcclUniqueIdDef {
    char internel[HCCL_UNIQUE_ID_BYTES];
};

typedef enum tagHcclRedOp {
    HCCL_REP_OP_SUM = 0,      /**< sum */
    HCCL_REP_OP_PROD = 1,     /**< prod */
    HCCL_REP_OP_MAX = 2,      /**< max */
    HCCL_REP_OP_MIN = 3,      /**< min */
    HCCL_REP_OP_RESERVED      /**< reserved */
} hcclRedOp_t;

typedef enum tagHcclDataType {
    HCCL_DATA_TYPE_INT8 = 0,  /**< int8 */
    HCCL_DATA_TYPE_INT = 1,   /**< int32 */
    HCCL_DATA_TYPE_HALF = 2,  /**< fp16 */
    HCCL_DATA_TYPE_FLOAT = 3, /**< fp32 */
    HCCL_DATA_TYPE_RESERVED   /**< reserved */
} hcclDataType_t;

typedef enum tagHcclResult {
    HCCL_SUCCESS = 0          /**< success */
} hcclResult_t;

/* handle to communicator */
typedef void *hcclComm_t;
typedef void *rtStream_t;

hcclResult_t HcclCommInitUniqueId(hcclComm_t* comm, u32 nranks, hcclUniqueId commId, u32 myrank);
hcclResult_t HcclGetUniqueId(hcclUniqueId* id);
hcclResult_t HcclAllReduce(void *inputPtr, void *outputPtr, u64 count, hcclDataType_t dataType,
                                  hcclRedOp_t op, hcclComm_t comm, rtStream_t stream);
hcclResult_t HcclBroadcast(void *ptr, u64 count, hcclDataType_t dataType, u32 root, hcclComm_t comm,
                                  rtStream_t stream);
hcclResult_t HcclCommDestroy(hcclComm_t comm);
}
