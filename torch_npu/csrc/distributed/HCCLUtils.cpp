#include <filesystem>
#include <fstream>
#include <string>

#include <torch/csrc/distributed/c10d/Utils.hpp>

#include "torch_npu/csrc/core/npu/interface/HcclInterface.h"
#include "torch_npu/csrc/distributed/HCCLUtils.hpp"


namespace c10d_npu {
bool isFileExists(const std::string& path)
{
    std::filesystem::path filePath(path);

    if (!filePath.is_absolute()) {
        TORCH_CHECK(false, "Path is not absolute.", DIST_ERROR(ErrCode::UNAVAIL))
        return false;
    }

    if (std::filesystem::exists(filePath) && std::filesystem::is_regular_file(filePath)) {
        return true;
    } else {
        return false;
    }
}

bool checkFilePathReadable(const std::string& file)
{
    std::filesystem::path filePath(file);

    if (!std::filesystem::exists(filePath)) {
        return false;
    }

    if (std::filesystem::is_symlink(filePath)) {
        return false;
    }

    if (!std::filesystem::is_regular_file(filePath)) {
        return false;
    }

    std::filesystem::perms perms = std::filesystem::status(filePath).permissions();
    if ((perms & std::filesystem::perms::owner_read) == std::filesystem::perms::owner_read) {
        return true;
    }
    return false;
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
    {at::ScalarType::Float8_e4m3fn, HCCL_DATA_TYPE_FP8E4M3},
    {at::ScalarType::Float8_e5m2, HCCL_DATA_TYPE_FP8E5M2},
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
    {HCCL_DATA_TYPE_FP8E4M3, "at::ScalarType::Float8_e4m3fn"},
    {HCCL_DATA_TYPE_FP8E5M2, "at::ScalarType::Float8_e5m2"},
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
    if (iter != kHcclDataTypeToStringMap.cend()) {
        return iter->second;
    } else {
        TORCH_NPU_WARN_ONCE("Can not serialize undefined hccl data type.");
        return "";
    }
}

bool isSupportHcclCommName()
{
    return at_npu::hccl::isHcclFeatureSupported(HcclCommConfigCapability::HCCL_COMM_CONFIG_COMM_NAME);
}

HCCLComm::HCCLComm(HcclComm hcclComm) : hcclComm_(hcclComm), hcclAsyncErr_(HCCL_SUCCESS),
    hcclCommType(0), p2pPeer(0) {}
    
HCCLComm::~HCCLComm()
{
    destroyHcclComm();
}

std::shared_ptr<HCCLComm> HCCLComm::create(
    int numRanks,
    int rank,
    HcclRootInfo& rootInfo)
{
    auto comm = std::make_shared<HCCLComm>();
    HCCL_CHECK_ERROR(HcclCommInitRootInfo(numRanks, &rootInfo, rank, &(comm->hcclComm_)));
    c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn([=]() ->void {comm->destroyHcclComm();},
                                                         c10_npu::ReleasePriority::PriorityMiddle);
    return comm;
}

std::shared_ptr<HCCLComm> HCCLComm::create_config(
    int numRanks,
    int rank,
    HcclRootInfo& rootInfo,
    HcclCommConfig* config)
{
    auto comm = std::make_shared<HCCLComm>();
    HCCL_CHECK_ERROR(hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config, &(comm->hcclComm_)));
    c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn([=]() ->void {comm->destroyHcclComm();},
                                                         c10_npu::ReleasePriority::PriorityMiddle);
    return comm;
}

std::shared_ptr<HCCLComm> HCCLComm::createGlobalHcclComm(
    const char *clusterInfo,
    uint32_t rank,
    HcclCommConfig* config)
{
    auto comm = std::make_shared<HCCLComm>();
    if (hcclCommInitClusterInfoConfig(clusterInfo, rank, config, &(comm->hcclComm_)) != HCCL_SUCCESS) {
        return nullptr;
    }
    c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn([=]() ->void {comm->destroyHcclComm();},
                                                         c10_npu::ReleasePriority::PriorityMiddle);
    return comm;
}

std::shared_ptr<HCCLComm> HCCLComm::createSubHcclComm(
    std::shared_ptr<HCCLComm> comm,
    uint32_t rankNum,
    uint32_t *rankIds,
    uint64_t subCommId,
    uint32_t subCommRankId,
    HcclCommConfig* config)
{
    auto subComm = std::make_shared<HCCLComm>();
    if (hcclCreateSubCommConfig(&(comm->hcclComm_), rankNum, rankIds, subCommId, subCommRankId,
        config, &(subComm->hcclComm_)) != HCCL_SUCCESS) {
        return nullptr;
    }
    c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn([=]() ->void {subComm->destroyHcclComm();},
                                                         c10_npu::ReleasePriority::PriorityMiddle);
    return subComm;
}

// Move constructable
HCCLComm::HCCLComm(HCCLComm&& other)
{
    std::swap(hcclComm_, other.hcclComm_);
    std::swap(hcclAsyncErr_, other.hcclAsyncErr_);
    std::swap(hcclCommType, other.hcclCommType);
    std::swap(p2pPeer, other.p2pPeer);
}

// Move assignable
HCCLComm& HCCLComm::operator=(HCCLComm&& other)
{
    std::swap(hcclComm_, other.hcclComm_);
    std::swap(hcclAsyncErr_, other.hcclAsyncErr_);
    std::swap(hcclCommType, other.hcclCommType);
    std::swap(p2pPeer, other.p2pPeer);
    return *this;
}

void HCCLComm::destroyHcclComm()
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (hcclComm_) {
        HcclCommDestroy(hcclComm_);
        hcclComm_ = nullptr;
    }
}

HcclResult HCCLComm::checkForHcclError()
{
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef ENABLE_HCCL_ERROR_CHECKING
    if (hcclAsyncErr_ != HCCL_SUCCESS) {
        return hcclAsyncErr_;
    }
    if (hcclComm_ != nullptr) {
        HcclResult result = hcclGetCommAsyncError(hcclComm_, &hcclAsyncErr_);
        if (result != HCCL_SUCCESS) {
            std::string temp_str = std::string("Failed to get HCCL error code: ") + std::to_string(result);
            const char* errmsg = temp_str.c_str();
            ASCEND_LOGE(errmsg);
            LOG(ERROR) << c10::str(errmsg);
            return result; // return this error result instead of hcclAsyncErr_
        }
    }
    return hcclAsyncErr_;
#else
    // Always return success, if error checks are disabled.
    return HCCL_SUCCESS;
#endif
}

void DebugInfoWriter::write(const std::string &hcclTrace)
{
    // Open a file for writing. The ios::binary flag is used to write data as
    // binary.
    std::ofstream file(filename_, std::ios::binary);

    // Check if the file was opened successfully.
    if (!file.is_open()) {
        LOG(ERROR) << "Error opening file for writing HCCLPG debug info: "
                   << filename_;
        return;
    }

    file.write(hcclTrace.data(), hcclTrace.size());
    LOG(INFO) << "Finished writing HCCLPG debug info to " << filename_;
}

DebugInfoWriter &DebugInfoWriter::getWriter(int rank)
{
    if (writer_ == nullptr) {
        std::string fileNamePrefix = c10d::getCvarString(
            {"TORCH_HCCL_DEBUG_INFO_TEMP_FILE"}, "/tmp/hccl_trace_rank_");
        // Using std::unique_ptr here to auto-delete the writer object
        // when the pointer itself is destroyed.
        std::unique_ptr<DebugInfoWriter> writerPtr(
            new DebugInfoWriter(fileNamePrefix, rank));
        DebugInfoWriter::registerWriter(std::move(writerPtr));
    }
    return *writer_;
}

void DebugInfoWriter::registerWriter(std::unique_ptr<DebugInfoWriter> writer)
{
    TORCH_CHECK_WITH(
        DistBackendError,
        !hasWriterRegistered_.load(),
        "debugInfoWriter already registered");
    hasWriterRegistered_.store(true);
    writer_ = std::move(writer);
}

std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);

struct HcclBufferNameKey {
    c10::DeviceIndex device_index;
    std::string name;
    bool operator<(const HcclBufferNameKey& other) const
    {
        if (device_index != other.device_index) {
            return device_index < other.device_index;
        }
        return name < other.name; // sort by name if device_index is the same
    }
};

struct HcclBufferNameStreamMap {
    std::map<HcclBufferNameKey, c10_npu::NPUStream> map;
    std::mutex mutex;
} g_BufferNameStreamMap = {};

c10::optional<c10_npu::NPUStream> getHcclStreamByBufferName(const std::string &name, c10::DeviceIndex device_index)
{
    std::unique_lock<std::mutex> lock(g_BufferNameStreamMap.mutex);
    auto &map = g_BufferNameStreamMap.map;
    auto it = map.find({device_index, name});
    if (it == map.end()) {
        return {};
    }
    return it->second;
}

bool setHcclStreamByBufferName(const std::string &name, c10::DeviceIndex device_index, c10_npu::NPUStream steam)
{
    HcclBufferNameKey key = {device_index, name};
    std::unique_lock<std::mutex> lock(g_BufferNameStreamMap.mutex);
    auto &map = g_BufferNameStreamMap.map;
    auto pair = map.insert({key, steam});
    return pair.second;
}

} // namespace c10d_npu
