#include "torch_npu/csrc/distributed/HCCLUtils.hpp"


namespace c10d_npu {
std::string getHcclErrorDetailStr(HcclResult error, c10::optional<std::string> processGroupFailureReason)
{
    // Prioritize failure reason provided by PG HCCL first, as it can abort
    // communicators when it encounters collective timeouts, etc.
    if (processGroupFailureReason != c10::nullopt) {
        return *processGroupFailureReason;
    }
    std::string interpret;

    switch (error) {
        case HCCL_E_REMOTE:
            interpret =
                "HCCL_E_REMOTE: A call failed possibly due to a network error or a remote process exiting prematurely.";
            break;
        default:
            interpret = "Unknown HCCL error!";
    }
    return interpret;
}

bool isFileExists(std::string& name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

// HCCL DataType mapping
std::map<at::ScalarType, HcclDataType> kScalarTypeToHcclDataType = {
    {at::kByte, HCCL_DATA_TYPE_UINT8},
    {at::kChar, HCCL_DATA_TYPE_INT8},
    {at::kShort, HCCL_DATA_TYPE_INT16},
    {at::kInt, HCCL_DATA_TYPE_INT32},
    {at::kLong, HCCL_DATA_TYPE_INT64},
    {at::kHalf, HCCL_DATA_TYPE_FP16},
    {at::kFloat, HCCL_DATA_TYPE_FP32},
    {at::kDouble, HCCL_DATA_TYPE_FP64},
    {at::kBool, HCCL_DATA_TYPE_UINT8},
    {at::kBFloat16, HCCL_DATA_TYPE_BFP16},
};

std::map<HcclDataType, std::string> kHcclDataTypeToStringMap = {
    {HCCL_DATA_TYPE_UINT8, "at::kByte/at::kBool"},
    {HCCL_DATA_TYPE_INT8, "at::kChar"},
    {HCCL_DATA_TYPE_INT16, "at::kShort"},
    {HCCL_DATA_TYPE_INT32, "at::kInt"},
    {HCCL_DATA_TYPE_INT64, "at::kLong"},
    {HCCL_DATA_TYPE_FP16, "at::kHalf"},
    {HCCL_DATA_TYPE_FP32, "at::kFloat"},
    {HCCL_DATA_TYPE_FP64, "at::kDouble"},
    {HCCL_DATA_TYPE_BFP16, "at::kBFloat16"},
};

// Helper function that gets the data type and issues error if not supported
HcclDataType getHcclDataType(at::ScalarType type)
{
    try {
        return kScalarTypeToHcclDataType.at(type);
    } catch (std::out_of_range& e) {
        throw std::runtime_error("Unsupported data type for HCCL process group" + DIST_ERROR(ErrCode::NOT_SUPPORT));
    }
}

std::string getHcclDataTypeSerialString(HcclDataType type)
{
    const auto& iter = kHcclDataTypeToStringMap.find(type);
    if (iter != kHcclDataTypeToStringMap.end()) {
        return iter->second;
    } else {
        TORCH_NPU_WARN_ONCE("Can not serialize undefined hccl data type.");
        return "";
    }
}

}
