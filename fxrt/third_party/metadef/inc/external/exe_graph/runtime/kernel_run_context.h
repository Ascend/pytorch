/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_KERNEL_RUN_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_KERNEL_RUN_CONTEXT_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*FreeCallback)(void*);

/**
 * Chain的底层数据结构，不要直接引用和操作此数据结构
 */
typedef struct {
  union {
    void* pointer;
    unsigned char inplace[sizeof(void*)];
  } data;
  FreeCallback deleter;
} AsyncAnyValue;

/**
 * KernelContext的底层数据结构，不要直接引用和操作此数据结构
 */
typedef struct {
  size_t input_size;
  size_t output_size;
  const void* compute_node_info;
  const void* kernel_extend_info;
  AsyncAnyValue** output_start; // todo delete this
  AsyncAnyValue* values[1];
} KernelRunContext;

#ifdef __cplusplus
}
#endif

#endif // METADEF_CXX_INC_EXE_GRAPH_KERNEL_RUN_CONTEXT_H_
