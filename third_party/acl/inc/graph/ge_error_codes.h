/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_
#define INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_

#include <cstdint>

namespace ge {
#if(defined(HOST_VISIBILITY)) && (defined(__GNUC__))
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_HOST_VISIBILITY
#endif
#if(defined(DEV_VISIBILITY)) && (defined(__GNUC__))
#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_DEV_VISIBILITY
#endif
#ifdef __GNUC__
#ifdef NO_METADEF_ABI_COMPATIABLE
#define ATTRIBUTED_DEPRECATED(replacement)
#define ATTRIBUTED_NOT_SUPPORT()
#else
#define ATTRIBUTED_DEPRECATED(replacement) __attribute__((deprecated("Please use " #replacement " instead.")))
#define ATTRIBUTED_NOT_SUPPORT() __attribute__((deprecated("The method will not be supported in the future.")))
#endif
#else
#ifdef NO_METADEF_ABI_COMPATIABLE
#define ATTRIBUTED_DEPRECATED(replacement)
#define ATTRIBUTED_NOT_SUPPORT()
#else
#define ATTRIBUTED_DEPRECATED(replacement) __declspec(deprecated("Please use " #replacement " instead."))
#define ATTRIBUTED_NOT_SUPPORT() __declspec(deprecated("The method will not be supported in the future."))
#endif
#endif

using graphStatus = uint32_t;
const graphStatus GRAPH_FAILED = 0xFFFFFFFF;
const graphStatus GRAPH_SUCCESS = 0;
const graphStatus GRAPH_NOT_CHANGED = 1343242304;

const graphStatus GRAPH_PARAM_INVALID = 50331649;
const graphStatus GRAPH_NODE_WITHOUT_CONST_INPUT = 50331648;
const graphStatus GRAPH_NODE_NEED_REPASS = 50331647;
const graphStatus GRAPH_INVALID_IR_DEF = 50331646;
const graphStatus OP_WITHOUT_IR_DATATYPE_INFER_RULE = 50331645;
const graphStatus GRAPH_PARAM_OUT_OF_RANGE = 50331644;

const graphStatus GRAPH_MEM_OPERATE_FAILED = 50331539;
const graphStatus GRAPH_NULL_PTR = 50331538;
const graphStatus GRAPH_MEMCPY_FAILED = 50331537;
const graphStatus GRAPH_MEMSET_FAILED = 50331536;

const graphStatus GRAPH_MATH_CAL_FAILED = 50331429;
const graphStatus GRAPH_ADD_OVERFLOW = 50331428;
const graphStatus GRAPH_MUL_OVERFLOW = 50331427;
const graphStatus GRAPH_RoundUp_Overflow = 50331426;
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_
