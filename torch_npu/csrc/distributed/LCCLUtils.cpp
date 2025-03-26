#include "LCCLUtils.hpp"

#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/FormatHelper.h"

namespace c10d_npu {

// LCCL DataType mapping
std::map<at::ScalarType, at_npu::lccl::LcclDataType> kScalarTypeToLcclDataType = {
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

std::map<at_npu::lccl::LcclDataType, std::string> kLcclDataTypeToStringMap = {
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

// LCCL unsupported ReduceOp
std::map<c10d::ReduceOp, std::string> unsupportedOp = {
    {c10d::ReduceOp::AVG, "AVG"},
    {c10d::ReduceOp::BAND, "BAND"},
    {c10d::ReduceOp::BOR, "BOR"},
    {c10d::ReduceOp::BXOR, "BXOR"}
};

// LCCL ReduceOp mapping
std::map<c10d::ReduceOp, at_npu::lccl::LcclReduceOp> lcclOp = {
    {c10d::ReduceOp::MIN, HCCL_REDUCE_MIN},
    {c10d::ReduceOp::MAX, HCCL_REDUCE_MAX},
    {c10d::ReduceOp::SUM, HCCL_REDUCE_SUM},
    {c10d::ReduceOp::PRODUCT, HCCL_REDUCE_PROD},
};

// Helper function that gets the data type and issues error if not supported
at_npu::lccl::LcclDataType getLcclDataType(at::ScalarType type)
{
    try {
        return kScalarTypeToLcclDataType.at(type);
    } catch (std::out_of_range& e) {
        throw std::runtime_error("Unsupported data type for LCCL process group" + DIST_ERROR(ErrCode::NOT_SUPPORT));
    }
}

std::string getLcclDataTypeSerialString(at_npu::lccl::LcclDataType type)
{
    const auto& iter = kLcclDataTypeToStringMap.find(type);
    if (iter != kLcclDataTypeToStringMap.cend()) {
        return iter->second;
    } else {
        TORCH_NPU_WARN_ONCE("Cannot serialize undefined LCCL data type.");
        return "";
    }
}

// AllGather & Broadcast support all data type, no need do more check.
void checkSupportedDataType(at_npu::lccl::LcclDataType type, std::string functionName)
{
    static std::set<at_npu::lccl::LcclDataType> supportedDataTypes = {
        HCCL_DATA_TYPE_INT8,
        HCCL_DATA_TYPE_INT16,
        HCCL_DATA_TYPE_INT32,
        HCCL_DATA_TYPE_FP16,
        HCCL_DATA_TYPE_FP32,
        HCCL_DATA_TYPE_BFP16,
        HCCL_DATA_TYPE_INT64};
    TORCH_CHECK(supportedDataTypes.count(type) != 0, "LCCL "+functionName+": Unsupported data type ",
        getLcclDataTypeSerialString(type), DIST_ERROR(ErrCode::NOT_SUPPORT));
}

at_npu::lccl::LcclReduceOp getLcclReduceOp(const c10d::ReduceOp reduceOp, at::Tensor& input)
{
    if (reduceOp == c10d::ReduceOp::SUM && input.scalar_type() == at::kBool) {
        // For bool tensors, map sum to max, which both represent a bitwise or.
        // This is to prevent overflow issues with sum, since we use uint8 to
        // represent a bool (see lcclDataType mapping).
        return HCCL_REDUCE_MAX;
    }

    if (unsupportedOp.find(reduceOp) != unsupportedOp.end()) {
        TORCH_CHECK(false, "Cannot use ReduceOp." + unsupportedOp[reduceOp] + " with LCCL",
            DIST_ERROR(ErrCode::NOT_SUPPORT));
    } else if (lcclOp.find(reduceOp) == lcclOp.end()) {
        TORCH_CHECK(false, "Unhandled ReduceOp", DIST_ERROR(ErrCode::NOT_FOUND));
    }
    return lcclOp[reduceOp];
}

// use tensor numel when the format is ACL_FORMAT_ND or ACL_FORMAT_NCHW
uint64_t getNumelForLCCL(const at::Tensor& self)
{
    aclFormat format = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.npu_format_;
    if (!at_npu::native::FormatHelper::IsBaseFormatType(format)) {
        if (self.storage().data_ptr().get() != self.data_ptr()) {
            TORCH_NPU_WARN_ONCE(
                "The storage data_ptr is different from tensor data_ptr."
                "Maybe this tensor is not suitable for LCCL.");
        }
        auto sizes = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.storage_sizes_;
        int64_t n = 1;
        for (auto s : sizes) {
            n *= s;
        }
        return n;
    } else {
        return self.numel();
    }
}

// Get the list of devices from list of tensors
std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors)
{
    std::vector<at::Device> res;
    res.reserve(tensors.size());
    for (auto& tensor : tensors) {
        res.push_back(tensor.device());
    }
    return res;
}

// Get the deviceList String from the list of devices
std::string getKeyFromDevices(const std::vector<at::Device>& devices)
{
    std::string deviceList;
    for (auto& device : devices) {
        if (deviceList.empty()) {
            deviceList = std::to_string(device.index());
        } else {
            deviceList += "," + std::to_string(device.index());
        }
    }
    return deviceList;
}

// Check that all `tensors' have the same type and shape and are distributed across distinct NPUs.
void checkTensors(const std::vector<at::Tensor>& tensors)
{
    if (tensors.size() == 0) {
        TORCH_CHECK(false, "Tensor list must be nonempty", DIST_ERROR(ErrCode::PARAM));
    }
    // HCCL support one NPU per process only
    if (tensors.size() != 1) {
        TORCH_CHECK(false, "Tensor list mustn't be larger than the number of available NPUs", DIST_ERROR(ErrCode::VALUE));
    }

    const auto& first = tensors.front();

    if (!torch_npu::utils::is_npu(first) || first.is_sparse()) {
        TORCH_CHECK(false, "Tensors must be NPU and dense", DIST_ERROR(ErrCode::TYPE));
    }
    if (!first.is_contiguous(first.suggest_memory_format())) {
        TORCH_CHECK(false, "Tensors must be contiguous", DIST_ERROR(ErrCode::TYPE));
    }
}

bool CheckTensorsSameSize(const std::vector<at::Tensor>& input_tensors)
{
    for (const auto& input_tensor : input_tensors) {
        if (!input_tensors[0].is_same_size(input_tensor)) {
            return false;
        }
    }
    return true;
}

std::vector<at::Tensor> castOriginFormat(const std::vector<at::Tensor>& inputTensors)
{
    std::vector<at::Tensor> inputTensors_;
    inputTensors_.resize(inputTensors.size());
    size_t index = 0;
    for (auto& tensor : inputTensors) {
        if (at_npu::native::FormatHelper::IsBaseFormatType(tensor)) {
            inputTensors_[index] = tensor;
        } else {
            auto origin_format = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.origin_format_;
            inputTensors_[index] = at_npu::native::npu_format_cast(tensor, origin_format);
        }
        index++;
    }
    return inputTensors_;
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> FlattenForScatterGather(std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other, size_t world_size)
{
    if (tensor_lists.size() != other.size()) {
        TORCH_CHECK(false, "Tensor list operands to scatter/gather must have the same length", DIST_ERROR(ErrCode::VALUE));
    }
    const auto num_devices = tensor_lists.size();

    std::vector<at::Tensor> flattened;
    flattened.resize(num_devices);

    for (auto i = size_t{}; i < num_devices; ++i) {
        if (tensor_lists[i].size() != world_size * num_devices) {
            TORCH_CHECK(false, "Tensor list input to scatter/gather must match number of collective participants",
                DIST_ERROR(ErrCode::PARAM));
        }

        // Only check device match for the first tensor in the list; the call to newLikeFlat() below will check the rest.
        if (tensor_lists[i].front().get_device() != other[i].get_device()) {
            TORCH_CHECK(false, "Corresponding input/output tensors to scatter/gather must all on the same device",
                DIST_ERROR(ErrCode::PARAM));
        }

        for (const auto& t : tensor_lists[i]) {
            if (t.numel() != other[i].numel()) {
                TORCH_CHECK(false, "All tensor operands to scatter/gather must have the same size",
                    DIST_ERROR(ErrCode::PARAM));
            }
        }
        // Flatten the tensors (from all ranks) into a single big tensor.
        flattened[i] = c10d::newLikeFlat(tensor_lists, i);
    }
    return flattened;
}

}
