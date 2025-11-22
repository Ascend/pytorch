/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GRAPH_C_TYPES_H_
#define INC_EXTERNAL_GRAPH_C_TYPES_H_
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    C_DT_FLOAT = 0,            // float type
    C_DT_FLOAT16 = 1,          // fp16 type
    C_DT_INT8 = 2,             // int8 type
    C_DT_INT32 = 3,            // int32 type
    C_DT_UINT8 = 4,            // uint8 type
    // reserved
    C_DT_INT16 = 6,            // int16 type
    C_DT_UINT16 = 7,           // uint16 type
    C_DT_UINT32 = 8,           // unsigned int32
    C_DT_INT64 = 9,            // int64 type
    C_DT_UINT64 = 10,          // unsigned int64
    C_DT_DOUBLE = 11,          // double type
    C_DT_BOOL = 12,            // bool type
    C_DT_STRING = 13,          // string type
    C_DT_DUAL_SUB_INT8 = 14,   // dual output int8 type
    C_DT_DUAL_SUB_UINT8 = 15,  // dual output uint8 type
    C_DT_COMPLEX64 = 16,       // complex64 type
    C_DT_COMPLEX128 = 17,      // complex128 type
    C_DT_QINT8 = 18,           // qint8 type
    C_DT_QINT16 = 19,          // qint16 type
    C_DT_QINT32 = 20,          // qint32 type
    C_DT_QUINT8 = 21,          // quint8 type
    C_DT_QUINT16 = 22,         // quint16 type
    C_DT_RESOURCE = 23,        // resource type
    C_DT_STRING_REF = 24,      // string ref type
    C_DT_DUAL = 25,            // dual output type
    C_DT_VARIANT = 26,         // dt_variant type
    C_DT_BF16 = 27,            // bf16 type
    C_DT_UNDEFINED = 28,       // Used to indicate a DataType field has not been set.
    C_DT_INT4 = 29,            // int4 type
    C_DT_UINT1 = 30,           // uint1 type
    C_DT_INT2 = 31,            // int2 type
    C_DT_UINT2 = 32,           // uint2 type
    C_DT_COMPLEX32 = 33,       // complex32 type
    C_DT_HIFLOAT8 = 34,        // hifloat8 type
    C_DT_FLOAT8_E5M2 = 35,     // float8_e5m2 type
    C_DT_FLOAT8_E4M3FN = 36,   // float8_e4m3fn type
    C_DT_FLOAT8_E8M0 = 37,     // float8_e8m0 type
    C_DT_FLOAT6_E3M2 = 38,     // float6_e3m2 type
    C_DT_FLOAT6_E2M3 = 39,     // float6_e2m3 type
    C_DT_FLOAT4_E2M1 = 40,     // float4_e2m1 type
    C_DT_FLOAT4_E1M2 = 41,     // float4_e1m2 type
    C_DT_MAX                   // Mark the boundaries of data types
} C_DataType;

typedef enum {
    C_FORMAT_NCHW = 0,   // NCHW
    C_FORMAT_NHWC,       // NHWC
    C_FORMAT_ND,         // Nd Tensor
    C_FORMAT_NC1HWC0,    // NC1HWC0
    C_FORMAT_FRACTAL_Z,  // FRACTAL_Z
    C_FORMAT_NC1C0HWPAD = 5,
    C_FORMAT_NHWC1C0,
    C_FORMAT_FSR_NCHW,
    C_FORMAT_FRACTAL_DECONV,
    C_FORMAT_C1HWNC0,
    C_FORMAT_FRACTAL_DECONV_TRANSPOSE = 10,
    C_FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
    C_FORMAT_NC1HWC0_C04,    // NC1HWC0, C0 is 4
    C_FORMAT_FRACTAL_Z_C04,  // FRACZ, C0 is 4
    C_FORMAT_CHWN,
    C_FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15,
    C_FORMAT_HWCN,
    C_FORMAT_NC1KHKWHWC0,  // KH,KW kernel h& kernel w maxpooling max output format
    C_FORMAT_BN_WEIGHT,
    C_FORMAT_FILTER_HWCK,  // filter input tensor format
    C_FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20,
    C_FORMAT_HASHTABLE_LOOKUP_KEYS,
    C_FORMAT_HASHTABLE_LOOKUP_VALUE,
    C_FORMAT_HASHTABLE_LOOKUP_OUTPUT,
    C_FORMAT_HASHTABLE_LOOKUP_HITS,
    C_FORMAT_C1HWNCoC0 = 25,
    C_FORMAT_MD,
    C_FORMAT_NDHWC,
    C_FORMAT_FRACTAL_ZZ,
    C_FORMAT_FRACTAL_NZ,
    C_FORMAT_NCDHW = 30,
    C_FORMAT_DHWCN,  // 3D filter input tensor format
    C_FORMAT_NDC1HWC0,
    C_FORMAT_FRACTAL_Z_3D,
    C_FORMAT_CN,
    C_FORMAT_NC = 35,
    C_FORMAT_DHWNC,
    C_FORMAT_FRACTAL_Z_3D_TRANSPOSE, // 3D filter(transpose) input tensor format
    C_FORMAT_FRACTAL_ZN_LSTM,
    C_FORMAT_FRACTAL_Z_G,
    C_FORMAT_RESERVED = 40,
    C_FORMAT_ALL,
    C_FORMAT_NULL,
    C_FORMAT_ND_RNN_BIAS,
    C_FORMAT_FRACTAL_ZN_RNN,
    C_FORMAT_NYUV = 45,
    C_FORMAT_NYUV_A,
    C_FORMAT_NCL,
    C_FORMAT_FRACTAL_Z_WINO,
    C_FORMAT_C1HWC0,
    C_FORMAT_FRACTAL_NZ_C0_16,
    C_FORMAT_FRACTAL_NZ_C0_32,
    // Add new formats definition here
    C_FORMAT_END,
    // FORMAT_MAX defines the max value of Format.
    // Any Format should not exceed the value of FORMAT_MAX.
    // ** Attention ** : FORMAT_MAX stands for the SPEC of enum Format and almost SHOULD NOT be used in code.
    //                   If you want to judge the range of Format, you can use FORMAT_END.
    C_FORMAT_MAX = 0xff
} C_Format;
#ifdef __cplusplus
}
#endif
#endif  // INC_EXTERNAL_GRAPH_C_TYPES_H_