// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


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

typedef enum tagHcclResult {
    HCCL_SUCCESS = 0          /**< success */
} hcclResult_t;

/* handle to communicator */
typedef void *hcclComm_t;
typedef void *rtStream_t;
typedef void *HcclComm;
typedef void *aclrtStream;

hcclResult_t HcclCommInitUniqueId(hcclComm_t* comm, u32 nranks, hcclUniqueId commId, u32 myrank);
hcclResult_t HcclGetUniqueId(hcclUniqueId* id);
hcclResult_t HcclGetCommName(hcclComm_t commHandle, char* commName);
hcclResult_t HcclAllReduce(void *inputPtr, void *outputPtr, u64 count, HcclDataType dataType,
                                  hcclRedOp_t op, hcclComm_t comm, rtStream_t stream);
hcclResult_t HcclBroadcast(void *ptr, u64 count, HcclDataType dataType, u32 root, hcclComm_t comm,
                                  rtStream_t stream);
hcclResult_t HcclCommDestroy(hcclComm_t comm);
hcclResult_t HcclScatter(void *sendBuf, void *recvBuf, u64 count, HcclDataType dataType, u32 root, HcclComm comm,
    aclrtStream stream);
}
