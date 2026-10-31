#include "torch_npu/csrc/custom_dtype/CustomDType.h"

#include <map>

namespace c10_npu {

const std::string CustomDataTypeToString(int64_t dType)
{
    if (!IsCustomDType(dType)) {
        return c10::toString(static_cast<at::ScalarType>(dType));
    }
    const std::map<const DType, const std::string>
        TYPE_TO_STRING_MAP = {
            {DType::FLOAT, "torch_npu.float32"},
            {DType::FLOAT16, "torch_npu.float16"},
            {DType::INT8, "torch_npu.int8"},
            {DType::INT32, "torch_npu.int32"},
            {DType::UINT8, "torch_npu.uint8"},
            {DType::INT16, "torch_npu.int16"},
            {DType::UINT16, "torch_npu.uint16"},
            {DType::UINT32, "torch_npu.uint32"},
            {DType::INT64, "torch_npu.int64"},
            {DType::UINT64, "torch_npu.uint64"},
            {DType::DOUBLE, "torch_npu.float64"},
            {DType::BOOL, "torch_npu.bool"},
            {DType::STRING, "torch_npu.string"},
            {DType::COMPLEX64, "torch_npu.complex64"},
            {DType::COMPLEX128, "torch_npu.complex128"},
            {DType::BF16, "torch_npu.bfloat16"},
            {DType::INT4, "torch_npu.int4"},
            {DType::UINT1, "torch_npu.uint1"},
            {DType::COMPLEX32, "torch_npu.complex32"},
            {DType::HIFLOAT8, "torch_npu.hifloat8"},
            {DType::FLOAT8_E5M2, "torch_npu.float8_e5m2"},
            {DType::FLOAT8_E4M3FN, "torch_npu.float8_e4m3fn"},
            {DType::FLOAT8_E8M0, "torch_npu.float8_e8m0fnu"},
            {DType::FLOAT6_E3M2, "torch_npu.float6_e3m2"},
            {DType::FLOAT6_E2M3, "torch_npu.float6_e2m3"},
            {DType::FLOAT4_E2M1, "torch_npu.float4_e2m1fn_x2"},
            {DType::FLOAT4_E1M2, "torch_npu.float4_e1m2fn_x2"}};

    const auto iter = TYPE_TO_STRING_MAP.find(static_cast<DType>(dType));
    return iter != TYPE_TO_STRING_MAP.end() ? iter->second : "Unknown dtype";
}

} // namespace c10_npu
