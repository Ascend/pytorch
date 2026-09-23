#ifndef OPS_ASCEND_HCCL_ADAPTER_H_
#define OPS_ASCEND_HCCL_ADAPTER_H_

#include "ops/ascend/hccl/hccl_plugin.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>

#include "hccl/hccl_types.h"

namespace fxrt::ops {
struct HcclTaskInfo {
  std::string private_def;
  int64_t workspace_size;
  int64_t stream_num;
};

struct HcclAllToAllVParams {
  std::vector<uint64_t> sendCounts;
  std::vector<uint64_t> sdispls;
  std::vector<uint64_t> recvCounts;
  std::vector<uint64_t> rdispls;
};

struct HcclAllGatherVParams {
  uint64_t sendCount;
  std::vector<uint64_t> recvCounts;
  std::vector<uint64_t> rdispls;
};

struct HcclReduceScatterVParams {
  std::vector<uint64_t> sendCounts;
  std::vector<uint64_t> sdispls;
  uint64_t recvCount;
};

struct HcclAllToAllParams {
  uint64_t sendCount;
  uint64_t recvCount;
};

enum HcclMode { kGraph, kPynative, kKernelByKernel };

class HcclAdapter {
 public:
  static HcclAdapter& GetInstance();

  // common
  bool InitHccl(uint32_t deviceId, std::string_view rankId, std::string_view rankFile, HcclMode hcclMode);
  bool InitHccl();
  uint32_t HcclGetCommConfigCapability();
  HcclResult HcclSetGlobalCommInfo(
      uint32_t masterIp,
      uint32_t masterPort,
      uint32_t totalRankSize,
      uint32_t nodeId,
      uint32_t localRankSize);
  HcclResult HcclCommInitClusterInfoConfig(
      const char* rankTable,
      uint32_t rankId,
      HcclCommConfig* config,
      HcclComm* hcclComm);
  HcclResult HcclCommInitRootInfoConfig(
      uint32_t nRanks,
      const HcclRootInfo* rootInfo,
      uint32_t rank,
      const HcclCommConfig* config,
      HcclComm* hcclComm);
  HcclResult HcclCreateSubCommConfig(
      HcclComm* globalComm,
      uint32_t rankSize,
      uint32_t* rankIds,
      uint64_t commId,
      uint32_t rankId,
      HcclCommConfig* config,
      HcclComm* hcclComm);
  bool FinalizeHccl();
  bool HcclWatchdogThread(HcclComm comm, std::string* errorInfo, bool* ret);
  const bool Inited() const {
    return initFlag_;
  }
  const HcclComm get_hccl_comm() const {
    return hcclComm_;
  }
  HcclResult HcclCreateGroup(const std::string& group, uint32_t rankNum, uint32_t* rankIds) const;
  HcclResult HcclDestroyGroup(const std::string& group) const;
  HcclResult HcclGetRankId(const std::string& group, uint32_t* rankId) const;
  HcclResult HcclGetRankSize(const std::string& group, uint32_t* rankSize) const;
  HcclResult HcclGetLocalRankId(const std::string& group, uint32_t* localRankId) const;
  HcclResult HcclGetLocalRankSize(const std::string& group, uint32_t* localRankSize) const;
  HcclResult HcclGetWorldRankFromGroupRank(const std::string& group, uint32_t localRank, uint32_t* worldRank) const;
  HcclResult HcclGetGroupRankFromWorldRank(uint32_t worldRank, const std::string& group, uint32_t* localRank) const;
  // for single op
  HcclResult HcclBroadcast(
      void* buf,
      uint64_t count,
      HcclDataType dataType,
      uint32_t root,
      aclrtStream stream,
      HcclComm hcclComm) const;
  HcclResult HcclAllReduce(
      void* sendBuf,
      void* recvBuf,
      uint64_t count,
      HcclDataType dataType,
      HcclReduceOp op,
      const aclrtStream stream,
      HcclComm hcclComm) const;
  HcclResult HcclReduce(
      void* sendBuf,
      void* recvBuf,
      uint64_t count,
      HcclDataType dataType,
      HcclReduceOp op,
      uint32_t root,
      const aclrtStream stream,
      HcclComm hcclComm) const;
  HcclResult HcclScatter(
      void* sendBuf,
      void* recvBuf,
      uint64_t count,
      HcclDataType dataType,
      uint32_t root,
      HcclComm hcclComm,
      aclrtStream stream) const;
  HcclResult HcclAllGather(
      void* sendBuf,
      void* recvBuf,
      uint64_t count,
      HcclDataType dataType,
      const aclrtStream stream,
      HcclComm hcclComm) const;
  HcclResult HcclReduceScatter(
      void* sendBuf,
      void* recvBuf,
      uint64_t count,
      HcclDataType dataType,
      HcclReduceOp op,
      const aclrtStream stream,
      HcclComm hcclComm) const;
  HcclResult HcclSend(
      void* sendBuf,
      uint64_t count,
      HcclDataType dataType,
      uint32_t destRank,
      const aclrtStream stream,
      HcclComm hcclComm) const;
  HcclResult HcclRecv(
      void* recvBuf,
      uint64_t count,
      HcclDataType dataType,
      uint32_t srcRank,
      const aclrtStream stream,
      HcclComm hcclComm) const;
  HcclResult HcclAlltoAllV(
      void* sendBuf,
      void* recvBuf,
      HcclAllToAllVParams params,
      HcclDataType dataType,
      const aclrtStream stream,
      HcclComm hcclComm) const;

  HcclResult HcclReduceScatterV(
      void* sendBuf,
      void* recvBuf,
      HcclReduceScatterVParams params,
      HcclDataType dataType,
      const HcclReduceOp op,
      const aclrtStream stream,
      HcclComm hcclComm) const;

  HcclResult HcclAllGatherV(
      void* sendBuf,
      void* recvBuf,
      HcclAllGatherVParams params,
      HcclDataType dataType,
      const aclrtStream stream,
      HcclComm hcclComm) const;

  HcclResult HcclAllToAll(
      void* sendBuf,
      void* recvBuf,
      HcclAllToAllParams params,
      HcclDataType dataType,
      const aclrtStream stream,
      HcclComm hcclComm) const;
  HcclResult HcclBarrier(const aclrtStream stream, HcclComm hcclComm) const;
  HcclResult HcclBatchISendIRecv(
      HcclSendRecvItem* sendRecvInfo,
      uint32_t itemNum,
      HcclComm hcclComm,
      aclrtStream stream) const;

  HcclResult HcclCommResume(HcclComm hcclComm) const;

  HcclResult HcclCommWorkingDevNicSet(HcclComm hcclComm, uint32_t* ranks, bool* useBackup, uint32_t nRanks);

  // Return whether using CM to initialize HCCL.
  bool UseHcclCM() const;
  static void AddCMEnvToHcclOption(std::map<std::string, std::string>* hcclOptMap);

  bool IsSameServer(const std::vector<uint32_t>& rankIds) const;

 private:
  HcclAdapter() = default;
  ~HcclAdapter() = default;
  void InitPlugin();
  void FinalizePlugin();

  bool InitKernelInfoStore(const std::map<std::string, std::string> options);
  bool FinalizeKernelInfoStore();

  bool InitHcclComm(std::string_view rankId, std::string_view rankFile);
  bool FinalizeHcclComm();

  bool InitHcclExec();
  bool FinalizeHcclExec();

  static std::string GetHcclModeString(HcclMode hcclMode);

  static bool IsSimulation();
  void* pluginHandle_ = nullptr;

  HcomDestroyFunObj hcomDestroy_ = nullptr;

  HcclGetCommConfigCapabilityFunObj getHcclCommConfigCapability_ = nullptr;
  HcclSetGlobalCommInfoFunObj setHcclGlobalCommInfo_ = nullptr;
  HcclCommInitClusterInfoFunObj initHcclComm_ = nullptr;
  HcclCommInitClusterInfoConfigFunObj initHcclGlobalCommRanktable_ = nullptr;
  HcclCommInitRootInfoConfigFunObj initHcclRootInfoConfig_ = nullptr;
  HcclCreateSubCommConfigFunObj initHcclSubCommRanktable_ = nullptr;
  HcclCommDestroyFunObj finalizeHcclComm_ = nullptr;
  HcclBroadcastFunObj launchHcclBroadcast_ = nullptr;
  HcclAllReduceFunObj launchHcclAllReduce_ = nullptr;
  HcclReduceFunObj launchHcclReduce_ = nullptr;
  HcclScatterFunObj launchHcclScatter_ = nullptr;
  HcclReduceScatterFunObj launchHcclReduceScatter_ = nullptr;
  HcclAllGatherFunObj launchHcclAllGather_ = nullptr;
  HcclSendFunObj launchHcclSend_ = nullptr;
  HcclRecvFunObj launchHcclRecv_ = nullptr;
  HcclBarrierFunObj launchHcclBarrier_ = nullptr;
  HcclGetRankIdFunObj singleOpHcclGetRankId_ = nullptr;
  HcclGetRankSizeFunObj singleOpHcclGetRankSize_ = nullptr;
  HcclAlltoAllVFunObj launchHcclAllToAllV_ = nullptr;
  HcclReduceScatterVFunObj launchHcclReduceScatterV_ = nullptr;
  HcclAllGatherVFunObj launchHcclAllGatherV_ = nullptr;
  HcclAlltoAllFunObj launchHcclAllToAll_ = nullptr;
  HcclBatchSendRecvFunObj launchHcclBatchISendIRecv_ = nullptr;
  HcclCommResumeFunObj launchHcclCommResume_ = nullptr;
  HcclGetCommAsyncErrorFunObj hcclGetCommAsyncError_ = nullptr;
  HcclGetErrorStringFunObj hcclGetErrorString_ = nullptr;
  HcomCreateGroupFunObj hcclCreateGroup_ = nullptr;
  HcomDestroyGroupFunObj hcclDestroyGroup_ = nullptr;
  HcomGetRankIdFunObj hcclGetRankId_ = nullptr;
  HcomGetRankSizeFunObj hcclGetRankSize_ = nullptr;
  HcomGetLocalRankIdFunObj hcclGetLocalRankId_ = nullptr;
  HcomGetLocalRankSizeFunObj hcclGetLocalRankSize_ = nullptr;
  HcomGetWorldRankFromGroupRankFunObj hcclGetWorldRankByGroupRank_ = nullptr;
  HcomGetGroupRankFromWorldRankFunObj hcclGetGroupRankByWorldRank_ = nullptr;
  HcclCommWorkingDevNicSetFunObj hcclCommWorkingDevNicSet_ = nullptr;

  HcomExecInitializeFunObj hcclExecInitialize_ = nullptr;
  HcomExecFinalizeFunObj hcclExecFinalize_ = nullptr;

  HcclComm hcclComm_ = nullptr;

  bool initFlag_ = false;
  bool initKernelInfoStore_ = false;
  bool initHcclExec_ = false;
  HcclMode hcclMode_ = HcclMode::kGraph;
  std::mutex initMutex_;
};
} // namespace fxrt::ops
#endif // OPS_ASCEND_HCCL_ADAPTER_H_
