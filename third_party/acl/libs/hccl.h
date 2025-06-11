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

typedef enum {
    HCCL_DATA_TYPE_INT8 = 0,    /**< int8 */
    HCCL_DATA_TYPE_INT16 = 1,   /**< int16 */
    HCCL_DATA_TYPE_INT32 = 2,   /**< int32 */
    HCCL_DATA_TYPE_FP16 = 3,    /**< fp16 */
    HCCL_DATA_TYPE_FP32 = 4,    /**< fp32 */
    HCCL_DATA_TYPE_INT64 = 5,   /**< int64 */
    HCCL_DATA_TYPE_UINT64 = 6,  /**< uint64 */
    HCCL_DATA_TYPE_UINT8 = 7,   /**< uint8 */
    HCCL_DATA_TYPE_UINT16 = 8,  /**< uint16 */
    HCCL_DATA_TYPE_UINT32 = 9,  /**< uint32 */
    HCCL_DATA_TYPE_FP64 = 10,   /**< fp64 */
    HCCL_DATA_TYPE_BFP16 = 11,  /**< bfp16 */
    HCCL_DATA_TYPE_RESERVED     /**< reserved */
} HcclDataType;

typedef enum {
    HCCL_REDUCE_SUM = 0,    /**< sum */
    HCCL_REDUCE_PROD = 1,   /**< prod */
    HCCL_REDUCE_MAX = 2,    /**< max */
    HCCL_REDUCE_MIN = 3,    /**< min */
    HCCL_REDUCE_RESERVED    /**< reserved */
} HcclReduceOp;

typedef enum tagHcclResult {
    HCCL_SUCCESS = 0          /**< success */
} hcclResult_t;

const u32 HCCL_ROOT_INFO_BYTES =  4108;
typedef struct HcclRootInfoDef {
    char internal[HCCL_ROOT_INFO_BYTES];
} HcclRootInfo;

typedef enum {
    HCCL_SEND = 0,
    HCCL_RECV = 1,
    HCCL_SEND_RECV_RESERVED
} HcclSendRecvType;

typedef struct HcclSendRecvItemDef {
    HcclSendRecvType sendRecvType;
    void *buf;
    u64 count;
    HcclDataType dataType;
    u32 remoteRank;
} HcclSendRecvItem;

typedef struct HcclCommConfigDef {
    char reserved[24];
    u32 hcclBufferSize;
    u32 hcclDeterministic;
} HcclCommConfig;

/* handle to communicator */
typedef void *hcclComm_t;
typedef void *rtStream_t;
typedef void *HcclComm;
typedef void *aclrtStream;

hcclResult_t HcclCommInitUniqueId(hcclComm_t* comm, u32 nranks, hcclUniqueId commId, u32 myrank);
hcclResult_t HcclGetUniqueId(hcclUniqueId* id);
hcclResult_t HcclAllReduce(void *inputPtr, void *outputPtr, u64 count, HcclDataType dataType,
                                  hcclRedOp_t op, hcclComm_t comm, rtStream_t stream);
hcclResult_t HcclBroadcast(void *ptr, u64 count, HcclDataType dataType, u32 root, hcclComm_t comm,
                                  rtStream_t stream);
hcclResult_t HcclCommDestroy(hcclComm_t comm);

hcclResult_t HcclReduceScatter(void *sendBuf, void *recvBuf, u64 recvCount, HcclDataType dataType,
                               HcclReduceOp op, HcclComm comm, aclrtStream stream);
hcclResult_t HcclCommInitRootInfo(u32 nRanks, const HcclRootInfo *rootInfo, u32 rank, HcclComm *comm);
hcclResult_t HcclCommInitRootInfoConfig(u32 nRanks, const HcclRootInfo *rootInfo, u32 rank, HcclCommConfig* config, HcclComm *comm);
hcclResult_t HcclCommInitClusterInfoConfig(const char *clusterInfo, u32 rank, HcclCommConfig *config, HcclComm *comm);
hcclResult_t HcclCreateSubCommConfig(HcclComm *comm, u32 rankNum, u32 *rankIds, u64 subCommId, u32 subCommRankId,
    HcclCommConfig *config, HcclComm *subComm);
hcclResult_t HcclGetCommName(HcclComm commHandle, char* commName);
hcclResult_t HcclAllGather(void *sendBuf, void *recvBuf, u64 sendCount, HcclDataType dataType, HcclComm comm,
                           aclrtStream stream);
hcclResult_t HcclRecv(void *recvBuf, u64 count, HcclDataType dataType, u32 srcRank, HcclComm comm, aclrtStream stream);
hcclResult_t HcclSend(void *sendBuf, u64 count, HcclDataType dataType, u32 destRank, HcclComm comm, aclrtStream stream);
hcclResult_t HcclGetRootInfo(HcclRootInfo *rootInfo);
hcclResult_t HcclGetCommAsyncError(hcclComm_t comm, hcclResult_t* asyncError);
hcclResult_t HcclScatter(void *sendBuf, void *recvBuf, u64 count, HcclDataType dataType, u32 root, HcclComm comm,
    aclrtStream stream);
hcclResult_t HcclBatchSendRecv(HcclSendRecvItemDef* sendRecvInfo, u32 itemNum, hcclComm_t comm, aclrtStream stream);
hcclResult_t HcclCommInitAll(u32 ndev, s32 *devices, hcclComm_t *comms);
hcclResult_t HcclCommResume(hcclComm_t comm);
hcclResult_t HcclCommWorkingDevNicSet(HcclComm comm, u32 *ranks, bool *useBackup, u32 nRanks);
}
