/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_BASE_RUNTIME_RUNTIME_ATTRS_DEF_H_
#define INC_EXTERNAL_BASE_RUNTIME_RUNTIME_ATTRS_DEF_H_
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  size_t attr_num;
  uint8_t reserved_[40]; // Reserved field, 32+8, do not directly use when only 8-byte left
  size_t offset[0];
} RuntimeAttrsDef;
#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_BASE_RUNTIME_RUNTIME_ATTRS_DEF_H_
