#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <torch/csrc/distributed/c10d/Utils.hpp>

#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/interface/HcclInterface.h"
#include "torch_npu/csrc/distributed/HCCLUtils.hpp"

namespace c10d_npu {
HcclResult hcclCommSymWinDeregister(void* handle);

bool isFileExists(const std::string& path) {
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

bool checkFilePathReadable(const std::string& file) {
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
    {at::ScalarType::UInt16, HCCL_DATA_TYPE_UINT16},
    {at::ScalarType::UInt32, HCCL_DATA_TYPE_UINT32},
    {at::ScalarType::UInt64, HCCL_DATA_TYPE_UINT64},
    {at::kDouble, HCCL_DATA_TYPE_FP64},
    {at::kBool, HCCL_DATA_TYPE_UINT8},
    {at::kBFloat16, HCCL_DATA_TYPE_BFP16},
    {at::ScalarType::Float8_e4m3fn, HCCL_DATA_TYPE_FP8E4M3},
    {at::ScalarType::Float8_e5m2, HCCL_DATA_TYPE_FP8E5M2},
    {at::ScalarType::Float8_e8m0fnu, HCCL_DATA_TYPE_FP8E8M0},
};

std::map<HcclDataType, std::string> kHcclDataTypeToStringMap = {
    {HCCL_DATA_TYPE_UINT8, "at::kByte/at::kBool"},
    {HCCL_DATA_TYPE_INT8, "at::kChar"},
    {HCCL_DATA_TYPE_INT16, "at::kShort"},
    {HCCL_DATA_TYPE_INT32, "at::kInt"},
    {HCCL_DATA_TYPE_UINT16, "at::ScalarType::UInt16"},
    {HCCL_DATA_TYPE_UINT32, "at::ScalarType::UInt32"},
    {HCCL_DATA_TYPE_UINT64, "at::ScalarType::UInt64"},
    {HCCL_DATA_TYPE_INT64, "at::kLong"},
    {HCCL_DATA_TYPE_FP16, "at::kHalf"},
    {HCCL_DATA_TYPE_FP32, "at::kFloat"},
    {HCCL_DATA_TYPE_FP64, "at::kDouble"},
    {HCCL_DATA_TYPE_BFP16, "at::kBFloat16"},
    {HCCL_DATA_TYPE_FP8E4M3, "at::ScalarType::Float8_e4m3fn"},
    {HCCL_DATA_TYPE_FP8E5M2, "at::ScalarType::Float8_e5m2"},
    {HCCL_DATA_TYPE_FP8E8M0, "at::ScalarType::Float8_e8m0fnu"},
};

// Helper function that gets the data type and issues error if not supported
HcclDataType getHcclDataType(at::ScalarType type) {
  // Other data types keep their existing mapping behavior without querying the SoC.
  TORCH_CHECK(
      type != at::ScalarType::Float8_e8m0fnu || c10_npu::GetSocVersion() == c10_npu::SocVersion::Ascend950,
      "HCCL float8_e8m0fnu communication is only supported on Ascend950 (Atlas A5).",
      DIST_ERROR(ErrCode::NOT_SUPPORT));
  try {
    return kScalarTypeToHcclDataType.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for HCCL process group" + DIST_ERROR(ErrCode::NOT_SUPPORT));
  }
}

std::string getHcclDataTypeSerialString(HcclDataType type) {
  const auto& iter = kHcclDataTypeToStringMap.find(type);
  if (iter != kHcclDataTypeToStringMap.cend()) {
    return iter->second;
  } else {
    TORCH_NPU_WARN_ONCE("Can not serialize undefined hccl data type.");
    return "";
  }
}

bool isSupportHcclCommName() {
  return at_npu::hccl::isHcclFeatureSupported(HcclCommConfigCapability::HCCL_COMM_CONFIG_COMM_NAME);
}

HCCLComm::HCCLComm(HcclComm hcclComm) : hcclComm_(hcclComm), hcclAsyncErr_(HCCL_SUCCESS), hcclCommType(0), p2pPeer(0) {}

HCCLComm::~HCCLComm() {
  destroyHcclComm();
}

std::shared_ptr<HCCLComm> HCCLComm::create(int numRanks, int rank, HcclRootInfo& rootInfo) {
  auto comm = std::make_shared<HCCLComm>();
  HCCL_CHECK_ERROR(hcclCommInitRootInfo(numRanks, &rootInfo, rank, &(comm->hcclComm_)));
  c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn(
      [=]() -> void { comm->destroyHcclComm(); }, c10_npu::ReleasePriority::PriorityMiddle);
  return comm;
}

std::shared_ptr<HCCLComm> HCCLComm::create_config(
    int numRanks,
    int rank,
    HcclRootInfo& rootInfo,
    HcclCommConfig* config) {
  auto comm = std::make_shared<HCCLComm>();
  HCCL_CHECK_ERROR(hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config, &(comm->hcclComm_)));
  c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn(
      [=]() -> void { comm->destroyHcclComm(); }, c10_npu::ReleasePriority::PriorityMiddle);
  return comm;
}

std::shared_ptr<HCCLComm> HCCLComm::create_scalable_config(
    int numRanks,
    int rank,
    const std::vector<HcclRootInfo>& rootInfoList,
    HcclCommConfig* config) {
  TORCH_CHECK(!rootInfoList.empty(), "The scalable HCCL root info list must not be empty.", DIST_ERROR(ErrCode::PARAM));
  auto comm = std::make_shared<HCCLComm>();
  HCCL_CHECK_ERROR(hcclCommInitRootInfoScalable(
      static_cast<uint32_t>(numRanks),
      static_cast<uint32_t>(rootInfoList.size()),
      rootInfoList.data(),
      static_cast<uint32_t>(rank),
      0, // nExtRoot is reserved for hierarchical roots and must currently be zero.
      config,
      &(comm->hcclComm_)));
  c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn(
      [=]() -> void { comm->destroyHcclComm(); }, c10_npu::ReleasePriority::PriorityMiddle);
  return comm;
}

std::shared_ptr<HCCLComm> HCCLComm::createGlobalHcclComm(
    const char* clusterInfo,
    uint32_t rank,
    HcclCommConfig* config) {
  auto comm = std::make_shared<HCCLComm>();
  if (hcclCommInitClusterInfoConfig(clusterInfo, rank, config, &(comm->hcclComm_)) != HCCL_SUCCESS) {
    return nullptr;
  }
  c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn(
      [=]() -> void { comm->destroyHcclComm(); }, c10_npu::ReleasePriority::PriorityMiddle);
  return comm;
}

std::shared_ptr<HCCLComm> HCCLComm::createSubHcclComm(
    std::shared_ptr<HCCLComm> comm,
    uint32_t rankNum,
    uint32_t* rankIds,
    uint64_t subCommId,
    uint32_t subCommRankId,
    HcclCommConfig* config) {
  auto subComm = std::make_shared<HCCLComm>();
  if (hcclCreateSubCommConfig(
          &(comm->hcclComm_), rankNum, rankIds, subCommId, subCommRankId, config, &(subComm->hcclComm_)) !=
      HCCL_SUCCESS) {
    return nullptr;
  }
  c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn(
      [=]() -> void { subComm->destroyHcclComm(); }, c10_npu::ReleasePriority::PriorityMiddle);
  return subComm;
}

// Move constructable
HCCLComm::HCCLComm(HCCLComm&& other) : HCCLComm() {
  std::lock_guard<std::mutex> lock(other.mutex_);
  std::swap(hcclComm_, other.hcclComm_);
  std::swap(hcclAsyncErr_, other.hcclAsyncErr_);
  std::swap(aborted_, other.aborted_);
  std::swap(registeredSegmentHandles_, other.registeredSegmentHandles_);
  std::swap(hcclCommType, other.hcclCommType);
  std::swap(p2pPeer, other.p2pPeer);
}

// Move assignable
HCCLComm& HCCLComm::operator=(HCCLComm&& other) {
  if (this == &other) {
    return *this;
  }
  {
    std::scoped_lock lock(mutex_, other.mutex_);
    std::swap(hcclComm_, other.hcclComm_);
    std::swap(hcclAsyncErr_, other.hcclAsyncErr_);
    std::swap(aborted_, other.aborted_);
    std::swap(registeredSegmentHandles_, other.registeredSegmentHandles_);
    std::swap(hcclCommType, other.hcclCommType);
    std::swap(p2pPeer, other.p2pPeer);
  }
  return *this;
}

void HCCLComm::destroyHcclComm() {
  HcclResult result;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (aborted_ || hcclComm_ == nullptr) {
      return;
    }
    result = hcclCommDestroy(hcclComm_);
    // Retire this object when discarding its handle, even if destruction fails.
    hcclComm_ = nullptr;
    aborted_ = true;
  }
  if (result != HCCL_SUCCESS) {
    LOG(ERROR) << "HcclCommDestroy failed with error " << result;
  }
}

void HCCLComm::releaseHcclCommRes() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (aborted_) {
    return;
  }
  // Deregister all symmetric windows before destroying the communicator.
  for (const auto& it : registeredSegmentHandles_) {
    HCCL_CHECK_ERROR(hcclCommSymWinDeregister(it.second.first), "hcclCommSymWinDeregister");
  }
  registeredSegmentHandles_.clear();
}

HcclResult HCCLComm::checkForHcclError() {
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
      ASCEND_LOGE("%s", errmsg);
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

HcclResult hcclCommSymWinRegister(HcclComm comm, void* addr, uint64_t size, void** handle, uint32_t flag);

HcclResult HCCLComm::registerSegment(void* ptr, size_t size, bool errorOnRereg, bool symmetric) {
  std::unique_lock<std::mutex> lock(mutex_);
  TORCH_CHECK(!aborted_, "HCCL communicator has been destroyed.", DIST_ERROR(ErrCode::UNAVAIL));
  void* handle = nullptr;
  if (registeredSegmentHandles_.count(ptr) > 0) {
    TORCH_CHECK(
        !errorOnRereg,
        "Segment with ptr ",
        ptr,
        " has already been registered on hcclComm_ ",
        hcclComm_,
        DIST_ERROR(ErrCode::UNAVAIL));
    return HCCL_SUCCESS;
  }
  TORCH_CHECK(
      symmetric,
      "HCCLComm::registerSegment: non-symmetric segment registration (symmetric=false) is not supported currently. ",
      "Please register the segment with symmetric=true. ",
      DIST_ERROR(ErrCode::NOT_SUPPORT));
  HCCL_CHECK_ERROR(hcclCommSymWinRegister(hcclComm_, ptr, size, &handle, 1), "hcclCommSymWinRegister");
  registeredSegmentHandles_[ptr] = std::make_pair(handle, symmetric);
  return HCCL_SUCCESS;
}

HcclResult HCCLComm::deregisterSegment(void* ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (aborted_) {
    // Cleanup is idempotent after destruction; do not deregister again or throw.
    return HCCL_SUCCESS;
  }
  auto it = registeredSegmentHandles_.find(ptr);
  TORCH_CHECK(
      it != registeredSegmentHandles_.end(),
      "Segment with ptr ",
      ptr,
      " is not registered on hcclComm_ ",
      hcclComm_,
      DIST_ERROR(ErrCode::UNAVAIL));
  void* handle = it->second.first;
  bool symmetric = it->second.second;
  TORCH_CHECK(
      symmetric,
      "HCCLComm::deregisterSegment: non-symmetric segment deregistration is not supported currently. "
      "Please deregister the segment registered with symmetric=true. ",
      DIST_ERROR(ErrCode::NOT_SUPPORT));
  HCCL_CHECK_ERROR(hcclCommSymWinDeregister(handle), "hcclCommSymWinDeregister");
  registeredSegmentHandles_.erase(it);
  return HCCL_SUCCESS;
}

std::string getHcclFlightRecorderDumpPrefix() {
#ifdef _WIN32
  const char* cacheHome = nullptr;
#else
  const char* cacheHome = std::getenv("XDG_CACHE_HOME");
#endif
  const std::string cacheRoot = cacheHome ? cacheHome : c10d::getCvarString({"HOME"}, "/tmp") + "/.cache";
  const auto cacheDirPath = std::filesystem::path(cacheRoot + "/torch");
  std::filesystem::create_directories(cacheDirPath);
  const auto defaultLocation = cacheDirPath / "comm_lib_trace_rank_";
  return c10d::getCvarString(
      {"TORCH_FR_DUMP_TEMP_FILE", "TORCH_HCCL_DEBUG_INFO_TEMP_FILE"}, defaultLocation.string().c_str());
}

void DebugInfoWriter::write(const std::string& trace) {
  std::string filename = filename_;
  if (enable_dynamic_filename_) {
    LOG(INFO) << "Writing Flight Recorder debug info to a dynamic file name";
    filename = c10d::getCvarString({"TORCH_FR_DUMP_TEMP_FILE"}, "");
  } else {
    LOG(INFO) << "Writing Flight Recorder debug info to a static file name";
  }
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    LOG(ERROR) << "Error opening file for writing Flight Recorder debug info: " << filename;
    return;
  }
  if (!file.write(trace.data(), static_cast<std::streamsize>(trace.size()))) {
    LOG(ERROR) << "Error writing Flight Recorder debug info to file: " << filename << " bad bit: " << file.bad();
    return;
  }
  file.flush();
  if (file.bad()) {
    LOG(ERROR) << "Error flushing Flight Recorder debug info: " << filename;
    return;
  }
  LOG(INFO) << "Finished writing Flight Recorder debug info to " << filename;
}

DebugInfoWriter& DebugInfoWriter::getWriter(int rank) {
  if (writer_ == nullptr) {
    const auto prefix = getHcclFlightRecorderDumpPrefix();
    const bool dynamic = c10d::getCvarBool({"TORCH_FR_DUMP_DYNAMIC_FILE_NAME"}, false);
    std::unique_ptr<DebugInfoWriter> writer(new DebugInfoWriter(prefix, rank, dynamic));
    registerWriter(std::move(writer));
  }
  return *writer_;
}

void DebugInfoWriter::registerWriter(std::unique_ptr<DebugInfoWriter> writer) {
  if (hasWriterRegistered_.load()) {
    TORCH_WARN_ONCE(
        "DebugInfoWriter has already been registered, and since we need the writer to stay "
        "outside ProcessGroup, user needs to ensure that this extra registration is indeed needed. "
        "And we will only use the last registered writer.");
  }
  hasWriterRegistered_.store(true);
  writer_ = std::move(writer);
}

std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);

struct HcclBufferNameKey {
  c10::DeviceIndex device_index;
  std::string name;
  bool operator<(const HcclBufferNameKey& other) const {
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

c10::optional<c10_npu::NPUStream> getHcclStreamByBufferName(const std::string& name, c10::DeviceIndex device_index) {
  std::unique_lock<std::mutex> lock(g_BufferNameStreamMap.mutex);
  auto& map = g_BufferNameStreamMap.map;
  auto it = map.find({device_index, name});
  if (it == map.end()) {
    return {};
  }
  return it->second;
}

bool setHcclStreamByBufferName(const std::string& name, c10::DeviceIndex device_index, c10_npu::NPUStream steam) {
  HcclBufferNameKey key = {device_index, name};
  std::unique_lock<std::mutex> lock(g_BufferNameStreamMap.mutex);
  auto& map = g_BufferNameStreamMap.map;
  auto pair = map.insert({key, steam});
  return pair.second;
}

} // namespace c10d_npu
