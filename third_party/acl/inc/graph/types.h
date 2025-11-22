/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GRAPH_TYPES_H_
#define INC_EXTERNAL_GRAPH_TYPES_H_

#include <atomic>
#include <memory>
#include <vector>
#include "c_types.h"

namespace ge {
using char_t = char;
using float32_t = float;
using float64_t = double;
using vector_bit_t = std::vector<bool>;

// set minimal range value as 0 to support empty tensor
static const int64_t SHAPE_RANGE_LOWER_LIMIT = 0;
static const int64_t UNKNOWN_DIM = -1;
static const int64_t UNKNOWN_DIM_NUM = -2;
#ifndef __NPU_DEVICE__
static const std::vector<int64_t> UNKNOWN_SHAPE = {-1};
static const std::vector<int64_t> UNKNOWN_RANK = {-2};
static const std::vector<int64_t> DUMMY_SHAPE = {-3};
#endif // __NPU_DEVICE__
// When data type unit is bit, this offset need to be added.
static constexpr int32_t kDataTypeSizeBitOffset = 1000;
static constexpr uint32_t kBitNumOfOneByte = 8U;
static constexpr uint32_t kBitThreeBytes = 24U;

#if defined(__GNUC__)
#ifndef GE_FUNC_HOST_VISIBILITY
#if defined(HOST_VISIBILITY)
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_HOST_VISIBILITY
#endif
#endif  // GE_FUNC_HOST_VISIBILITY

#ifndef GE_FUNC_DEV_VISIBILITY
#if defined(DEV_VISIBILITY)
#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_DEV_VISIBILITY
#endif
#endif  // GE_FUNC_DEV_VISIBILITY

#ifndef WEAK_SYMBOL
#define WEAK_SYMBOL __attribute__((weak))
#endif

#ifndef FORMAT_PRINTF
#define FORMAT_PRINTF(format_idx, first_arg) __attribute__((format(printf, (format_idx), (first_arg))))
#endif
#else
#ifndef GE_FUNC_HOST_VISIBILITY
#define GE_FUNC_HOST_VISIBILITY
#endif

#ifndef GE_FUNC_DEV_VISIBILITY
#define GE_FUNC_DEV_VISIBILITY
#endif

#ifndef WEAK_SYMBOL
#define WEAK_SYMBOL
#endif

#ifndef FORMAT_PRINTF
#define FORMAT_PRINTF(format_idx, first_arg)
#endif
#endif  // defined(__GNUC__)

enum DataType {
    DT_FLOAT = ::C_DT_FLOAT,
    DT_FLOAT16 = ::C_DT_FLOAT16,
    DT_INT8 = ::C_DT_INT8,
    DT_INT32 = ::C_DT_INT32,
    DT_UINT8 = ::C_DT_UINT8,
    DT_INT16 = ::C_DT_INT16,
    DT_UINT16 = ::C_DT_UINT16,
    DT_UINT32 = ::C_DT_UINT32,
    DT_INT64 = ::C_DT_INT64,
    DT_UINT64 = ::C_DT_UINT64,
    DT_DOUBLE = ::C_DT_DOUBLE,
    DT_BOOL = ::C_DT_BOOL,
    DT_STRING = ::C_DT_STRING,
    DT_DUAL_SUB_INT8 = ::C_DT_DUAL_SUB_INT8,
    DT_DUAL_SUB_UINT8 = ::C_DT_DUAL_SUB_UINT8,
    DT_COMPLEX64 = ::C_DT_COMPLEX64,
    DT_COMPLEX128 = ::C_DT_COMPLEX128,
    DT_QINT8 = ::C_DT_QINT8,
    DT_QINT16 = ::C_DT_QINT16,
    DT_QINT32 = ::C_DT_QINT32,
    DT_QUINT8 = ::C_DT_QUINT8,
    DT_QUINT16 = ::C_DT_QUINT16,
    DT_RESOURCE = ::C_DT_RESOURCE,
    DT_STRING_REF = ::C_DT_STRING_REF,
    DT_DUAL = ::C_DT_DUAL,
    DT_VARIANT = ::C_DT_VARIANT,
    DT_BF16 = ::C_DT_BF16,
    DT_UNDEFINED = ::C_DT_UNDEFINED,
    DT_INT4 = ::C_DT_INT4,
    DT_UINT1 = ::C_DT_UINT1,
    DT_INT2 = ::C_DT_INT2,
    DT_UINT2 = ::C_DT_UINT2,
    DT_COMPLEX32 = ::C_DT_COMPLEX32,
    DT_HIFLOAT8 = ::C_DT_HIFLOAT8,
    DT_FLOAT8_E5M2 = ::C_DT_FLOAT8_E5M2,
    DT_FLOAT8_E4M3FN = ::C_DT_FLOAT8_E4M3FN,
    DT_FLOAT8_E8M0 = ::C_DT_FLOAT8_E8M0,
    DT_FLOAT6_E3M2 = ::C_DT_FLOAT6_E3M2,
    DT_FLOAT6_E2M3 = ::C_DT_FLOAT6_E2M3,
    DT_FLOAT4_E2M1 = ::C_DT_FLOAT4_E2M1,
    DT_FLOAT4_E1M2 = ::C_DT_FLOAT4_E1M2,
    DT_MAX = ::C_DT_MAX,
};

// used for data type of DT_STRING
struct StringHead {
    int64_t addr;  // the addr of string
    int64_t len;   // the length of string
};

inline int GetSizeByDataType(DataType data_type) {
    static int data_type_size[DT_MAX] = {
        4,                           // DT_FLOAT = 0,             float type
        2,                           // DT_FLOAT16 = 1,           fp16 type
        1,                           // DT_INT8 = 2,              int8 type
        4,                           // DT_INT32 = 3,             int32 type
        1,                           // DT_UINT8 = 4,             uint8 type
        -1,                          // reserved
        2,                           // DT_INT16 = 6,             int16 type
        2,                           // DT_UINT16 = 7,            uint16 type
        4,                           // DT_UINT32 = 8,            unsigned int32
        8,                           // DT_INT64 = 9,             int64 type
        8,                           // DT_UINT64 = 10,           unsigned int64
        8,                           // DT_DOUBLE = 11,           double type
        1,                           // DT_BOOL = 12,             bool type
        -1,                          // DT_STRING = 13,           string type
        1,                           // DT_DUAL_SUB_INT8 = 14,    dual output int8 type
        1,                           // DT_DUAL_SUB_UINT8 = 15,   dual output uint8 type
        8,                           // DT_COMPLEX64 = 16,        complex64 type
        16,                          // DT_COMPLEX128 = 17,       complex128 type
        1,                           // DT_QINT8 = 18,            qint8 type
        2,                           // DT_QINT16 = 19,           qint16 type
        4,                           // DT_QINT32 = 20,           qint32 type
        1,                           // DT_QUINT8 = 21,           quint8 type
        2,                           // DT_QUINT16 = 22,          quint16 type
        8,                           // DT_RESOURCE = 23,         resource type
        -1,                          // DT_STRING_REF = 24,       string ref type
        5,                           // DT_DUAL = 25,             dual output type (float + int8)
        8,                           // DT_VARIANT                variant type
        2,                           // DT_BF16 = 27,             bf16 type
        -1,                          // DT_UNDEFINED = 28         Used to indicate a DataType field has not been set.
        kDataTypeSizeBitOffset + 4,  // DT_INT4 = 29,             int4 type
        kDataTypeSizeBitOffset + 1,  // DT_UINT1 = 30,            uint1 type
        kDataTypeSizeBitOffset + 2,  // DT_INT2 = 31,             int2 type
        kDataTypeSizeBitOffset + 2,  // DT_UINT2 = 32,            uint2 type
        4,                           // DT_COMPLEX32 = 33,        complex32 type
        1,                           // DT_HIFLOAT8,              hifloat8 type
        1,                           // DT_FLOAT8_E5M2,           float8_e5m2 type
        1,                           // DT_FLOAT8_E4M3FN,         float8_e4m3fn type
        1,                           // DT_FLOAT8_E8M0,           float8_e8m0 type
        kDataTypeSizeBitOffset + 6,  // DT_FLOAT6_E3M2,           float6_e3m2 type, 6bit
        kDataTypeSizeBitOffset + 6,  // DT_FLOAT6_E2M3,           float6_e2m3 type, 6bit
        kDataTypeSizeBitOffset + 4,  // DT_FLOAT4_E2M1,           float4_e2m1 type, 4bit
        kDataTypeSizeBitOffset + 4,  // DT_FLOAT4_E1M2,           float4_e1m2 type, 4bit
                                    // DT_MAX
    };
    if ((data_type < 0) || (data_type >= DT_MAX)) {
        return -1;
    }
    return data_type_size[data_type];
}

/// @brief Calculates the length in bytes based on the DataType and the number of elements.
/// @param element_count
/// @param data_type
/// @return
int64_t GetSizeInBytes(int64_t element_count, DataType data_type);

enum Format {
    FORMAT_NCHW = ::C_FORMAT_NCHW,
    FORMAT_NHWC = ::C_FORMAT_NHWC,
    FORMAT_ND = ::C_FORMAT_ND,
    FORMAT_NC1HWC0 = ::C_FORMAT_NC1HWC0,
    FORMAT_FRACTAL_Z = ::C_FORMAT_FRACTAL_Z,
    FORMAT_NC1C0HWPAD = ::C_FORMAT_NC1C0HWPAD,
    FORMAT_NHWC1C0 = ::C_FORMAT_NHWC1C0,
    FORMAT_FSR_NCHW = ::C_FORMAT_FSR_NCHW,
    FORMAT_FRACTAL_DECONV = ::C_FORMAT_FRACTAL_DECONV,
    FORMAT_C1HWNC0 = ::C_FORMAT_C1HWNC0,
    FORMAT_FRACTAL_DECONV_TRANSPOSE = ::C_FORMAT_FRACTAL_DECONV_TRANSPOSE,
    FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS = ::C_FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS,
    FORMAT_NC1HWC0_C04 = ::C_FORMAT_NC1HWC0_C04,
    FORMAT_FRACTAL_Z_C04 = ::C_FORMAT_FRACTAL_Z_C04,
    FORMAT_CHWN = ::C_FORMAT_CHWN,
    FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = ::C_FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS,
    FORMAT_HWCN = ::C_FORMAT_HWCN,
    FORMAT_NC1KHKWHWC0 = ::C_FORMAT_NC1KHKWHWC0,
    FORMAT_BN_WEIGHT = ::C_FORMAT_BN_WEIGHT,
    FORMAT_FILTER_HWCK = ::C_FORMAT_FILTER_HWCK,
    FORMAT_HASHTABLE_LOOKUP_LOOKUPS = ::C_FORMAT_HASHTABLE_LOOKUP_LOOKUPS,
    FORMAT_HASHTABLE_LOOKUP_KEYS = ::C_FORMAT_HASHTABLE_LOOKUP_KEYS,
    FORMAT_HASHTABLE_LOOKUP_VALUE = ::C_FORMAT_HASHTABLE_LOOKUP_VALUE,
    FORMAT_HASHTABLE_LOOKUP_OUTPUT = ::C_FORMAT_HASHTABLE_LOOKUP_OUTPUT,
    FORMAT_HASHTABLE_LOOKUP_HITS = ::C_FORMAT_HASHTABLE_LOOKUP_HITS,
    FORMAT_C1HWNCoC0 = ::C_FORMAT_C1HWNCoC0,
    FORMAT_MD = ::C_FORMAT_MD,
    FORMAT_NDHWC = ::C_FORMAT_NDHWC,
    FORMAT_FRACTAL_ZZ = ::C_FORMAT_FRACTAL_ZZ,
    FORMAT_FRACTAL_NZ = ::C_FORMAT_FRACTAL_NZ,
    FORMAT_NCDHW = ::C_FORMAT_NCDHW,
    FORMAT_DHWCN = ::C_FORMAT_DHWCN,
    FORMAT_NDC1HWC0 = ::C_FORMAT_NDC1HWC0,
    FORMAT_FRACTAL_Z_3D = ::C_FORMAT_FRACTAL_Z_3D,
    FORMAT_CN = ::C_FORMAT_CN,
    FORMAT_NC = ::C_FORMAT_NC,
    FORMAT_DHWNC = ::C_FORMAT_DHWNC,
    FORMAT_FRACTAL_Z_3D_TRANSPOSE = ::C_FORMAT_FRACTAL_Z_3D_TRANSPOSE,
    FORMAT_FRACTAL_ZN_LSTM = ::C_FORMAT_FRACTAL_ZN_LSTM,
    FORMAT_FRACTAL_Z_G = ::C_FORMAT_FRACTAL_Z_G,
    FORMAT_RESERVED = ::C_FORMAT_RESERVED,
    FORMAT_ALL = ::C_FORMAT_ALL,
    FORMAT_NULL = ::C_FORMAT_NULL,
    FORMAT_ND_RNN_BIAS = ::C_FORMAT_ND_RNN_BIAS,
    FORMAT_FRACTAL_ZN_RNN = ::C_FORMAT_FRACTAL_ZN_RNN,
    FORMAT_NYUV = ::C_FORMAT_NYUV,
    FORMAT_NYUV_A = ::C_FORMAT_NYUV_A,
    FORMAT_NCL = ::C_FORMAT_NCL,
    FORMAT_FRACTAL_Z_WINO = ::C_FORMAT_FRACTAL_Z_WINO,
    FORMAT_C1HWC0 = ::C_FORMAT_C1HWC0,
    FORMAT_FRACTAL_NZ_C0_16 = ::C_FORMAT_FRACTAL_NZ_C0_16,
    FORMAT_FRACTAL_NZ_C0_32 = ::C_FORMAT_FRACTAL_NZ_C0_32,
    FORMAT_END = ::C_FORMAT_END,
    FORMAT_MAX = ::C_FORMAT_MAX,
};


/// Get format from primary and sub-format,
/// in bits field:
/// ---------------------------------------------
/// |   4bits  |   4bits   |   2 bytes  | 1 byte |
/// |----------|-----------|------------|--------|
/// | reserved | c0_format | sub-format | format |
/// ---------------------------------------------
/// @param primary_format
/// @param sub_format
/// @param c0_format
/// @return
inline int32_t GetFormatFromSub(int32_t primary_format, int32_t sub_format)
{
    return static_cast<int32_t>((static_cast<uint32_t>(primary_format) & 0xffU) |
                                ((static_cast<uint32_t>(sub_format) & 0xffffU) << kBitNumOfOneByte));
}

inline int32_t GetFormatFromC0(int32_t format, int32_t c0_format)
{
    return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xffffffU) |
                                ((static_cast<uint32_t>(c0_format) & 0xfU) << kBitThreeBytes));
}

inline int32_t GetFormatFromSubAndC0(int32_t primary_format, int32_t sub_format, int32_t c0_format)
{
    return static_cast<int32_t>((static_cast<uint32_t>(primary_format) & 0xffU) |
                                ((static_cast<uint32_t>(sub_format) & 0xffffU) << kBitNumOfOneByte) |
                                ((static_cast<uint32_t>(c0_format) & 0xfU) << kBitThreeBytes));
}

inline int32_t GetPrimaryFormat(int32_t format)
{
    return static_cast<int32_t>(static_cast<uint32_t>(format) & 0xffU);
}

inline int32_t GetSubFormat(int32_t format)
{
    return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xffff00U) >> kBitNumOfOneByte);
}

inline bool HasSubFormat(int32_t format)
{
    return GetSubFormat(format) > 0;
}

inline bool HasC0Format(int32_t format)
{
    return ((static_cast<uint32_t>(format) & 0xf000000U) >> kBitThreeBytes) > 0;
}

inline int32_t GetC0Format(int32_t format)
{
    return static_cast<int32_t>((static_cast<uint32_t>(format) & 0xf000000U) >> kBitThreeBytes);
}

inline int64_t GetC0Value(int32_t format)
{
    if (!HasC0Format(format)) {
        return -1;
    }
    return static_cast<int64_t>(1 <<
        (static_cast<int32_t>((static_cast<uint32_t>(format) & 0xf000000U) >> kBitThreeBytes) - 1));
}

// for unknown shape op type
enum UnknowShapeOpType {
  DEPEND_IN_SHAPE    = 1,  // op out shape get by input shape
  DEPEND_CONST_VALUE = 2,  // op out shape get by const op value
  DEPEND_SHAPE_RANGE = 3,  // op out shape get by range
  DEPEND_COMPUTE     = 4   // op out shape get by totally computing
};

struct TensorDescInfo {
  Format format_ = FORMAT_RESERVED;  // tbe op register support format
  DataType dataType_ = DT_UNDEFINED; // tbe op register support datatype
};

enum DeviceType {
  NPU = 0,
  CPU = 1,
};

enum Placement {
  kPlacementHost = 0,     // host data addr
  kPlacementDevice = 1,   // device data addr
  kPlacementEnd,
};

///
/// @brief Get a format name from enum
/// @param format
/// @return
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
const char_t *GetFormatName(Format format);

class TensorTypeImpl;
struct TensorType {
    explicit TensorType(DataType dt);

    TensorType(const std::initializer_list<DataType> &initial_types);

    static TensorType ALL()
    {
        return TensorType{DT_BOOL,   DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                          DT_INT32,  DT_INT64,      DT_INT8,      DT_QINT16, DT_QINT32, DT_QINT8,   DT_QUINT16,
                          DT_QUINT8, DT_RESOURCE,   DT_STRING,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                          DT_BF16, DT_COMPLEX32};
    }

    static TensorType QuantifiedType() { return TensorType{DT_QINT16, DT_QINT32, DT_QINT8, DT_QUINT16, DT_QUINT8}; }

    static TensorType OrdinaryType()
    {
        return TensorType{DT_BOOL,  DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                          DT_INT32, DT_INT64,      DT_INT8,      DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                          DT_BF16, DT_COMPLEX32};
    }

    static TensorType BasicType()
    {
        return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,
                          DT_INT32,      DT_INT64,     DT_INT8,   DT_QINT16, DT_QINT32,  DT_QINT8,
                          DT_QUINT16,    DT_QUINT8,    DT_UINT16, DT_UINT32, DT_UINT64,  DT_UINT8,
                          DT_BF16, DT_COMPLEX32};
    }

    static TensorType NumberType()
    {
        return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,  DT_INT32,  DT_INT64,
                          DT_INT8,       DT_QINT32,    DT_QINT8,  DT_QUINT8, DT_UINT16,  DT_UINT32, DT_UINT64, DT_UINT8,
                          DT_BF16, DT_COMPLEX32};
    }

    static TensorType RealNumberType()
    {
        return TensorType{DT_DOUBLE, DT_FLOAT,  DT_FLOAT16, DT_INT16,  DT_INT32, DT_INT64,
                          DT_INT8,   DT_UINT16, DT_UINT32,  DT_UINT64, DT_UINT8, DT_BF16};
    }

    static TensorType ComplexDataType() { return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_COMPLEX32}; }

    static TensorType IntegerDataType()
    {
        return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8};
    }

    static TensorType SignedDataType() { return TensorType{DT_INT16, DT_INT32, DT_INT64, DT_INT8}; }

    static TensorType UnsignedDataType() { return TensorType{DT_UINT16, DT_UINT32, DT_UINT64, DT_UINT8}; }

    static TensorType FloatingDataType() { return TensorType{DT_DOUBLE, DT_FLOAT, DT_FLOAT16}; }

    static TensorType IndexNumberType() { return TensorType{DT_INT32, DT_INT64}; }

    static TensorType UnaryDataType()
    {
        return TensorType{DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_COMPLEX32};
    }

    static TensorType FLOAT() { return TensorType{DT_FLOAT, DT_FLOAT16, DT_BF16}; }

    std::shared_ptr<TensorTypeImpl> tensor_type_impl_;
};

struct ListTensorType {
    explicit ListTensorType(const TensorType &type) : tensor_type(type) {}
    TensorType tensor_type;
};

class Promote {
public:
  friend class PromoteImpl;
  Promote(const std::initializer_list<const char *> &syms);
  std::vector<const char *> Syms() const;

  Promote(const Promote &other) = delete;
  Promote &operator=(const Promote &other) = delete;

  Promote(Promote &&other) noexcept;
  Promote &operator=(Promote &&other) noexcept;

private:
  std::shared_ptr<void> data_;
};
}  // namespace ge

namespace domi {
enum class ImplyType : unsigned int {
  BUILDIN = 0,  // Built in operator, normally executed by OME
  TVM,          // Compile to TVM bin file for execution
  CUSTOM,       // User defined calculation logic, executed by CPU
  AI_CPU,       // AICPU
  CCE,          // Cce
  GELOCAL,      // GE local, do node need execute by device
  HCCL,         // Hccl
  INVALID = 0xFFFFFFFF,
};
using char_t = ge::char_t;
using float32_t = ge::float32_t;
using float64_t = ge::float64_t;
}  // namespace domi

#endif  // INC_EXTERNAL_GRAPH_TYPES_H_
