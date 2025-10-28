/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_TYPES_H
#define SHMEM_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Data op engine type.
*/
enum data_op_engine_type_t {
    SHMEM_DATA_OP_MTE = 0x01,
    SHMEM_DATA_OP_SDMA = 0x02,
    SHMEM_DATA_OP_ROCE = 0x04,
};

/**@} */ // end of group_typedef

#ifdef __cplusplus
}
#endif

#endif /* SHMEM_TYPES_H */