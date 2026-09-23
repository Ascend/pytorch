#include "ops/ascend/hccl/hccl_adapter.h"
#include <dlfcn.h>
#include <map>
#include <algorithm>
#include <sstream>

#include "hccl/hccl.h"

static constexpr const auto kHcclPluginFileName = "libhccl.so";

#define CHECK_SYMBOL_NULL(symbol)                                                   \
  if ((symbol) == nullptr) {                                                        \
    RT_GLOG(ERROR) << #symbol << " is null, hccl has not been inited, do nothing."; \
    return HcclResult::HCCL_E_RESERVED;                                             \
  }

namespace fxrt::ops {
const char kDefaultGroup[] = "__default_group";
constexpr uint32_t kDeviceNumOfServer = 8;

HcclAdapter& HcclAdapter::GetInstance() {
  static HcclAdapter instance;
  return instance;
}

void HcclAdapter::InitPlugin() {
  if (pluginHandle_ != nullptr) {
    return;
  }
#ifndef ENABLE_ASAN
  pluginHandle_ = dlopen(kHcclPluginFileName, RTLD_DEEPBIND | RTLD_NOW | RTLD_LOCAL);
#else
  pluginHandle_ = dlopen(kHcclPluginFileName, RTLD_NOW | RTLD_LOCAL);
#endif
  if (pluginHandle_ == nullptr) {
    RT_GLOG(EXCEPTION) << "Dlopen " << kHcclPluginFileName << " failed, result = " << GetDlErrorMsg();
  }

  launchHcclAllReduce_ = DlsymFuncObj(HcclAllReduce, pluginHandle_);
  launchHcclReduceScatter_ = DlsymFuncObj(HcclReduceScatter, pluginHandle_);
  launchHcclAllGather_ = DlsymFuncObj(HcclAllGather, pluginHandle_);
  launchHcclAllToAll_ = DlsymFuncObj(HcclAlltoAll, pluginHandle_);
  launchHcclAllToAllV_ = DlsymFuncObj(HcclAlltoAllV, pluginHandle_);
}

void HcclAdapter::FinalizePlugin() {
  if (pluginHandle_ == nullptr) {
    return;
  }
  setHcclGlobalCommInfo_ = nullptr;
  initHcclRootInfoConfig_ = nullptr;
  initHcclGlobalCommRanktable_ = nullptr;
  initHcclSubCommRanktable_ = nullptr;
  getHcclCommConfigCapability_ = nullptr;
  initHcclComm_ = nullptr;
  finalizeHcclComm_ = nullptr;
  launchHcclBroadcast_ = nullptr;
  launchHcclAllReduce_ = nullptr;
  launchHcclReduce_ = nullptr;
  launchHcclScatter_ = nullptr;
  launchHcclReduceScatter_ = nullptr;
  launchHcclAllGather_ = nullptr;
  launchHcclSend_ = nullptr;
  launchHcclRecv_ = nullptr;
  launchHcclBarrier_ = nullptr;
  launchHcclBatchISendIRecv_ = nullptr;
  hcclCreateGroup_ = nullptr;
  hcclDestroyGroup_ = nullptr;
  hcclGetRankId_ = nullptr;
  hcclGetLocalRankId_ = nullptr;
  hcclGetLocalRankSize_ = nullptr;
  hcclGetWorldRankByGroupRank_ = nullptr;
  hcclGetGroupRankByWorldRank_ = nullptr;
  hcclGetRankSize_ = nullptr;
  hcclCommWorkingDevNicSet_ = nullptr;
  launchHcclAllToAllV_ = nullptr;
  launchHcclReduceScatterV_ = nullptr;
  launchHcclAllGatherV_ = nullptr;
  launchHcclCommResume_ = nullptr;
  hcomDestroy_ = nullptr;
  (void)dlclose(pluginHandle_);
  pluginHandle_ = nullptr;
}

std::string HcclAdapter::GetHcclModeString(HcclMode hcclMode) {
  static std::map<HcclMode, std::string> kHcclModeString = {
      {HcclMode::kGraph, "GE_MODE"},
      {HcclMode::kPynative, "PYNATIVE_MODE"},
      {HcclMode::kKernelByKernel, "KERNEL_BY_KERNEL_MODE"}};
  return kHcclModeString.at(hcclMode);
}

bool HcclAdapter::InitHccl() {
  std::lock_guard<std::mutex> lock(initMutex_);
  if (initFlag_) {
    return true;
  }
  InitPlugin();

  initFlag_ = true;
  RT_VLOG(VL_OPS) << "Init hccl adapter success.";
  return true;
}

bool HcclAdapter::HcclWatchdogThread(HcclComm comm, std::string* errorInfo, bool* disable) {
  if (!initFlag_) {
    RT_VLOG(VL_OPS) << "Hccl has never been inited, skip.";
    return true;
  }
  CHECK_IF_NULL(disable);
  if (hcclGetCommAsyncError_ == nullptr) {
    RT_VLOG(VL_OPS) << "Hccl has never been inited, skip.";
    return true;
  }
  if (hcclGetErrorString_ == nullptr) {
    RT_VLOG(VL_OPS) << "Hccl has never been inited, skip.";
    return true;
  }
  HcclResult hcclAsyncError;
  auto ret = hcclGetCommAsyncError_(comm, &hcclAsyncError);
  if (ret != HCCL_SUCCESS) {
    RT_VLOG(VL_OPS) << "Call HcclGetCommAsyncError failed, close watchdog.";
    *disable = true;
    return true;
  }
  if (hcclAsyncError != HCCL_SUCCESS) {
    std::ostringstream oss;
    oss << "Hccl get comm async error failed, error code is: " << hcclAsyncError
        << ", detail info: " << hcclGetErrorString_(hcclAsyncError);
    *errorInfo = oss.str();
    return false;
  }
  return true;
}

// The HcomExec queue and the kernel info store are not used by the
// kernel-by-kernel collectives, so their finalizers are no-ops here.
bool HcclAdapter::FinalizeHcclExec() {
  return true;
}

bool HcclAdapter::FinalizeKernelInfoStore() {
  return true;
}

bool HcclAdapter::FinalizeHccl() {
  std::lock_guard<std::mutex> lock(initMutex_);
  RT_VLOG(VL_OPS) << "Start destroy hccl adapter for " << GetHcclModeString(hcclMode_);
  if (!initFlag_) {
    RT_VLOG(VL_OPS) << "Hccl has never been inited, skip.";
    return true;
  }
  (void)FinalizeHcclExec();
  (void)FinalizeKernelInfoStore();
  (void)FinalizeHcclComm();
  if (hcomDestroy_ != nullptr) {
    hcomDestroy_();
  }
  FinalizePlugin();
  initFlag_ = false;
  RT_VLOG(VL_OPS) << "Destroy hccl adapter success.";
  return true;
}

HcclResult HcclAdapter::HcclBroadcast(
    void* buf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t root,
    aclrtStream stream,
    HcclComm hcclComm) const {
  HcclResult ret = launchHcclBroadcast_(buf, count, dataType, root, hcclComm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclAllReduce(
    void* sendBuf,
    void* recvBuf,
    uint64_t count,
    HcclDataType dataType,
    const HcclReduceOp op,
    const aclrtStream stream,
    HcclComm hcclComm) const {
  HcclResult ret = launchHcclAllReduce_(sendBuf, recvBuf, count, dataType, op, hcclComm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclReduce(
    void* sendBuf,
    void* recvBuf,
    uint64_t count,
    HcclDataType dataType,
    HcclReduceOp op,
    uint32_t root,
    const aclrtStream stream,
    HcclComm hcclComm) const {
  HcclResult ret = launchHcclReduce_(sendBuf, recvBuf, count, dataType, op, root, hcclComm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclScatter(
    void* sendBuf,
    void* recvBuf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t root,
    const aclrtStream stream,
    HcclComm hcclComm) const {
  HcclResult ret = launchHcclScatter_(sendBuf, recvBuf, count, dataType, root, hcclComm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclReduceScatter(
    void* sendBuf,
    void* recvBuf,
    uint64_t count,
    HcclDataType dataType,
    const HcclReduceOp op,
    const aclrtStream stream,
    HcclComm hcclComm) const {
  HcclResult ret = launchHcclReduceScatter_(sendBuf, recvBuf, count, dataType, op, hcclComm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclAllGather(
    void* sendBuf,
    void* recvBuf,
    uint64_t count,
    HcclDataType dataType,
    const aclrtStream stream,
    HcclComm hcclComm) const {
  CHECK_SYMBOL_NULL(launchHcclAllGather_);
  CHECK_IF_NULL(hcclComm);
  CHECK_IF_NULL(sendBuf);
  CHECK_IF_NULL(recvBuf);
  HcclResult ret = launchHcclAllGather_(sendBuf, recvBuf, count, dataType, hcclComm, stream);
  return ret;
}

HcclResult HcclAdapter::HcclSend(
    void* sendBuf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t destRank,
    const aclrtStream stream,
    HcclComm hcclComm) const {
  HcclResult ret = launchHcclSend_(sendBuf, count, dataType, destRank, hcclComm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclRecv(
    void* recvBuf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t srcRank,
    const aclrtStream stream,
    HcclComm hcclComm) const {
  HcclResult ret = launchHcclRecv_(recvBuf, count, dataType, srcRank, hcclComm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclBarrier(const aclrtStream stream, HcclComm hcclComm) const {
  return launchHcclBarrier_(hcclComm, stream);
}

HcclResult HcclAdapter::HcclBatchISendIRecv(
    HcclSendRecvItem* sendRecvInfo,
    uint32_t itemNum,
    HcclComm hcclComm,
    aclrtStream stream) const {
  HcclResult ret = launchHcclBatchISendIRecv_(sendRecvInfo, itemNum, hcclComm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclCommResume(HcclComm hcclComm) const {
  if (launchHcclCommResume_ == nullptr) {
    RT_GLOG(EXCEPTION) << "Dynamically load HcclCommResume failed.";
  }
  return launchHcclCommResume_(hcclComm);
}

uint32_t HcclAdapter::HcclGetCommConfigCapability() {
  CHECK_IF_NULL(getHcclCommConfigCapability_);
  return getHcclCommConfigCapability_();
}

HcclResult HcclAdapter::HcclSetGlobalCommInfo(
    uint32_t masterIp,
    uint32_t masterPort,
    uint32_t totalRankSize,
    uint32_t nodeId,
    uint32_t localRankSize) {
  if (setHcclGlobalCommInfo_ == nullptr) {
    setHcclGlobalCommInfo_ = DlsymAscendFuncObj(HcclSetGlobalCommInfo, pluginHandle_);
    if (setHcclGlobalCommInfo_ == nullptr) {
      RT_VLOG(VL_OPS) << "Func HcclSetGlobalCommInfo is not supported in CANN package.";
      return HCCL_E_NOT_SUPPORT;
    }
  }
  return setHcclGlobalCommInfo_(masterIp, masterPort, totalRankSize, nodeId, localRankSize);
}

HcclResult HcclAdapter::HcclCommInitClusterInfoConfig(
    const char* rankTable,
    uint32_t rankId,
    HcclCommConfig* config,
    HcclComm* hcclComm) {
  if (initHcclGlobalCommRanktable_ == nullptr) {
    initHcclGlobalCommRanktable_ = DlsymFuncObj(HcclCommInitClusterInfoConfig, pluginHandle_);
  }
  return initHcclGlobalCommRanktable_(rankTable, rankId, config, hcclComm);
}
HcclResult HcclAdapter::HcclCommInitRootInfoConfig(
    uint32_t nRanks,
    const HcclRootInfo* rootInfo,
    uint32_t rank,
    const HcclCommConfig* config,
    HcclComm* hcclComm) {
  if (initHcclRootInfoConfig_ == nullptr) {
    initHcclRootInfoConfig_ = DlsymFuncObj(HcclCommInitRootInfoConfig, pluginHandle_);
    if (initHcclRootInfoConfig_ == nullptr) {
      // new api in CANN C20
      return HcclCommInitRootInfo(nRanks, rootInfo, rank, hcclComm);
    }
  }

  return initHcclRootInfoConfig_(nRanks, rootInfo, rank, config, hcclComm);
}

HcclResult HcclAdapter::HcclCreateSubCommConfig(
    HcclComm* globalComm,
    uint32_t rankSize,
    uint32_t* rankIds,
    uint64_t commId,
    uint32_t rankId,
    HcclCommConfig* config,
    HcclComm* hcclComm) {
  if (initHcclSubCommRanktable_ == nullptr) {
    initHcclSubCommRanktable_ = DlsymFuncObj(HcclCreateSubCommConfig, pluginHandle_);
  }
  return initHcclSubCommRanktable_(globalComm, rankSize, rankIds, commId, rankId, config, hcclComm);
}

bool HcclAdapter::InitHcclComm(std::string_view rankId, std::string_view rankFile) {
  RT_VLOG(VL_OPS) << "Start init hccl comm.";
  int rankIdI = -1;
  try {
    rankIdI = std::stoi(rankId.data());
  } catch (std::invalid_argument&) {
    RT_GLOG(EXCEPTION) << "Invalid rank id env:" << rankId;
  }
  if (rankIdI < 0) {
    RT_GLOG(ERROR) << "rank_id cannot be negative";
    return false;
  }
  CHECK_IF_NULL(initHcclComm_);
  auto hcclResult = initHcclComm_(rankFile.data(), rankIdI, &hcclComm_);
  if (hcclResult != HCCL_SUCCESS) {
    RT_GLOG(ERROR) << "HcclCommInitClusterInfo failed, ret:" << hcclResult;
    return false;
  }
  RT_VLOG(VL_OPS) << "InitHcclComm success";
  return true;
}

bool HcclAdapter::FinalizeHcclComm() {
  RT_VLOG(VL_OPS) << "Start finalize hccl comm.";
  if (hcclComm_ == nullptr) {
    return true;
  }

  CHECK_IF_NULL(finalizeHcclComm_);
  auto hcclResult = finalizeHcclComm_(hcclComm_);
  if (hcclResult != HCCL_SUCCESS) {
    RT_GLOG(ERROR) << "HcclComm destroy failed, ret:" << hcclResult;
    return false;
  }
  hcclComm_ = nullptr;
  RT_VLOG(VL_OPS) << "HcclComm destroy success";
  return true;
}

HcclResult HcclAdapter::HcclCreateGroup(const std::string& group, uint32_t rankNum, uint32_t* rankIds) const {
  CHECK_SYMBOL_NULL(hcclCreateGroup_);
  return hcclCreateGroup_(group.c_str(), rankNum, rankIds);
}
HcclResult HcclAdapter::HcclDestroyGroup(const std::string& group) const {
  CHECK_SYMBOL_NULL(hcclDestroyGroup_);
  return hcclDestroyGroup_(group.c_str());
}

HcclResult HcclAdapter::HcclGetRankId(const std::string& group, uint32_t* rankId) const {
  if (hcclMode_ != HcclMode::kGraph) {
    CHECK_SYMBOL_NULL(singleOpHcclGetRankId_);
    return singleOpHcclGetRankId_(hcclComm_, rankId);
  } else {
    CHECK_SYMBOL_NULL(hcclGetRankId_);
    return hcclGetRankId_(group.c_str(), rankId);
  }
}

HcclResult HcclAdapter::HcclGetRankSize(const std::string& group, uint32_t* rankSize) const {
  if (hcclMode_ != HcclMode::kGraph) {
    CHECK_SYMBOL_NULL(singleOpHcclGetRankSize_);
    return singleOpHcclGetRankSize_(hcclComm_, rankSize);
  } else {
    CHECK_SYMBOL_NULL(hcclGetRankSize_);
    return hcclGetRankSize_(group.c_str(), rankSize);
  }
}

HcclResult HcclAdapter::HcclGetLocalRankId(const std::string& group, uint32_t* localRankId) const {
  CHECK_SYMBOL_NULL(hcclGetLocalRankId_);
  return hcclGetLocalRankId_(group.c_str(), localRankId);
}

HcclResult HcclAdapter::HcclGetLocalRankSize(const std::string& group, uint32_t* localRankSize) const {
  if (hcclMode_ != HcclMode::kGraph) {
    RT_GLOG(ERROR) << "The pynative mode doesn't support get local rank size.";
    return HCCL_E_NOT_SUPPORT;
  } else {
    CHECK_SYMBOL_NULL(hcclGetLocalRankSize_);
    return hcclGetLocalRankSize_(group.c_str(), localRankSize);
  }
}

HcclResult HcclAdapter::HcclGetWorldRankFromGroupRank(const std::string& group, uint32_t localRank, uint32_t* worldRank)
    const {
  if (hcclMode_ != HcclMode::kGraph) {
    RT_GLOG(ERROR) << "The pynative mode doesn't support get world rank by group rank.";
    return HCCL_E_NOT_SUPPORT;
  } else {
    CHECK_SYMBOL_NULL(hcclGetWorldRankByGroupRank_);
    return hcclGetWorldRankByGroupRank_(group.c_str(), localRank, worldRank);
  }
}

HcclResult HcclAdapter::HcclGetGroupRankFromWorldRank(uint32_t worldRank, const std::string& group, uint32_t* localRank)
    const {
  if (hcclMode_ != HcclMode::kGraph) {
    RT_GLOG(ERROR) << "The pynative mode doesn't support get group rank by world rank.";
    return HCCL_E_NOT_SUPPORT;
  } else {
    CHECK_SYMBOL_NULL(hcclGetGroupRankByWorldRank_);
    return hcclGetGroupRankByWorldRank_(worldRank, group.c_str(), localRank);
  }
}

HcclResult HcclAdapter::HcclCommWorkingDevNicSet(HcclComm hcclComm, uint32_t* ranks, bool* useBackup, uint32_t nRanks) {
  if (hcclCommWorkingDevNicSet_ == nullptr) {
    hcclCommWorkingDevNicSet_ = DlsymFuncObj(HcclCommWorkingDevNicSet, pluginHandle_);
  }
  CHECK_SYMBOL_NULL(hcclCommWorkingDevNicSet_);
  return hcclCommWorkingDevNicSet_(hcclComm, ranks, useBackup, nRanks);
}

bool HcclAdapter::UseHcclCM() const {
  return false;
}

HcclResult HcclAdapter::HcclAlltoAllV(
    void* sendBuf,
    void* recvBuf,
    HcclAllToAllVParams params,
    HcclDataType dataType,
    aclrtStream stream,
    HcclComm hcclComm) const {
  CHECK_SYMBOL_NULL(launchHcclAllToAllV_);
  CHECK_IF_NULL(hcclComm);
  HcclResult ret = launchHcclAllToAllV_(
      sendBuf,
      params.sendCounts.data(),
      params.sdispls.data(),
      dataType,
      recvBuf,
      params.recvCounts.data(),
      params.rdispls.data(),
      dataType,
      hcclComm,
      stream);

  return ret;
}

HcclResult HcclAdapter::HcclReduceScatterV(
    void* sendBuf,
    void* recvBuf,
    HcclReduceScatterVParams params,
    HcclDataType dataType,
    const HcclReduceOp op,
    const aclrtStream stream,
    HcclComm hcclComm) const {
  CHECK_SYMBOL_NULL(launchHcclReduceScatterV_);
  CHECK_IF_NULL(hcclComm);
  HcclResult ret = launchHcclReduceScatterV_(
      sendBuf,
      params.sendCounts.data(),
      params.sdispls.data(),
      recvBuf,
      params.recvCount,
      dataType,
      op,
      hcclComm,
      stream);
  return ret;
}

HcclResult HcclAdapter::HcclAllGatherV(
    void* sendBuf,
    void* recvBuf,
    HcclAllGatherVParams params,
    HcclDataType dataType,
    const aclrtStream stream,
    HcclComm hcclComm) const {
  CHECK_SYMBOL_NULL(launchHcclAllGatherV_);
  CHECK_IF_NULL(hcclComm);
  HcclResult ret = launchHcclAllGatherV_(
      sendBuf, params.sendCount, recvBuf, params.recvCounts.data(), params.rdispls.data(), dataType, hcclComm, stream);
  return ret;
}

HcclResult HcclAdapter::HcclAllToAll(
    void* sendBuf,
    void* recvBuf,
    HcclAllToAllParams params,
    HcclDataType dataType,
    aclrtStream stream,
    HcclComm hcclComm) const {
  CHECK_SYMBOL_NULL(launchHcclAllToAll_);
  CHECK_IF_NULL(hcclComm);

  HcclResult ret =
      launchHcclAllToAll_(sendBuf, params.sendCount, dataType, recvBuf, params.recvCount, dataType, hcclComm, stream);

  return ret;
}

bool HcclAdapter::IsSameServer(const std::vector<uint32_t>& rankIds) const {
  auto minIter = min_element(rankIds.begin(), rankIds.end());
  uint32_t min = (minIter != rankIds.end()) ? *minIter : 0;
  auto maxIter = max_element(rankIds.begin(), rankIds.end());
  uint32_t max = (maxIter != rankIds.end()) ? *maxIter : 0;
  return ((max - min < kDeviceNumOfServer) && (min / kDeviceNumOfServer == max / kDeviceNumOfServer));
}

} // namespace fxrt::ops
