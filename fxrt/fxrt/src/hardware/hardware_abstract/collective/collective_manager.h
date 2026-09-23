#ifndef FXRT_SRC_HARDWARE_CLUSTER_COLLECTIVE_MANAGER_H_
#define FXRT_SRC_HARDWARE_CLUSTER_COLLECTIVE_MANAGER_H_

#include <string>
#include <memory>
#include <vector>
#include <queue>
#include <atomic>
#include <utility>
#include <thread>
#include <functional>
#include <mutex>
#include <unordered_map>

#include "common/logger.h"
#include "common/visible.h"
#include "hardware/hardware_abstract/collective/communication_group.h"

namespace fxrt {
namespace collective {
class DA_API CollectiveManager {
 public:
  ~CollectiveManager();
  static CollectiveManager& Instance();

  void SetGlobalRankId(uint32_t globalRankId);
  void SetGlobalRankSize(uint32_t globalRankSize);
  void SetLocalRankId(uint32_t localRankId);

  uint32_t global_rank_id() const;
  uint32_t local_rank_id() const;
  uint32_t global_rank_size() const;

  bool CreateCommunicationGroup(
      const std::string& groupName,
      const std::vector<uint32_t>& groupRanks,
      uint32_t groupRank,
      int64_t communicator);
  bool IsGroupExist(const std::string& groupName);
  std::shared_ptr<CommunicationGroup> GetCommunicationGroup(const std::string& groupName);

  uint32_t GetGroupRank(const std::string& groupName);
  uint32_t GetGroupSize(const std::string& groupName);

 private:
  CollectiveManager();
  CollectiveManager(const CollectiveManager&) = delete;
  CollectiveManager& operator=(const CollectiveManager&) = delete;

  uint32_t globalRankId;
  uint32_t localRankId;
  uint32_t globalRankSize;
  std::vector<uint32_t> globalGroupRanks;
  std::unordered_map<std::string, std::shared_ptr<CommunicationGroup>> communicationGroups;
};

} // namespace collective
} // namespace fxrt
#endif // FXRT_SRC_HARDWARE_CLUSTER_COLLECTIVE_MANAGER_H_
