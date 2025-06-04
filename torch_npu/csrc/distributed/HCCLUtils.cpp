#include <filesystem>
#include <fstream>

#include <torch/csrc/distributed/c10d/Utils.hpp>

#include "torch_npu/csrc/core/npu/interface/HcclInterface.h"
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

HCCLComm::HCCLComm(HcclComm hcclComm) : hcclComm_(hcclComm), hcclAsyncErr_(HCCL_SUCCESS) {}
    
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
}

// Move assignable
HCCLComm& HCCLComm::operator=(HCCLComm&& other)
{
    std::swap(hcclComm_, other.hcclComm_);
    std::swap(hcclAsyncErr_, other.hcclAsyncErr_);
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
        C10D_HCCL_CHECK(hcclGetCommAsyncError(hcclComm_, &hcclAsyncErr_));
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
        hasWriterRegistered_.load() == false,
        "debugInfoWriter already registered");
    hasWriterRegistered_.store(true);
    writer_ = std::move(writer);
}

std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);
} // namespace c10d_npu
