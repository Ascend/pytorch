#include "hardware/hardware_abstract/collective/collective_manager.h"

namespace fxrt {
namespace collective {

CollectiveManager::CollectiveManager() : globalRankId(0), localRankId(0), globalRankSize(0) {}

CollectiveManager::~CollectiveManager() {}

CollectiveManager& CollectiveManager::Instance() {
  static CollectiveManager instance;
  return instance;
}

bool CollectiveManager::CreateCommunicationGroup(
    const std::string& groupName,
    const std::vector<uint32_t>& groupRanks,
    uint32_t groupRank,
    int64_t communicator) {
  if (communicationGroups.find(groupName) != communicationGroups.end()) {
    return false;
  }
  communicationGroups[groupName] = std::make_shared<CommunicationGroup>(groupName, groupRanks, groupRank, communicator);
  return true;
}

bool CollectiveManager::IsGroupExist(const std::string& groupName) {
  return communicationGroups.find(groupName) != communicationGroups.end();
}

std::shared_ptr<CommunicationGroup> CollectiveManager::GetCommunicationGroup(const std::string& groupName) {
  if (communicationGroups.find(groupName) == communicationGroups.end()) {
    RT_GLOG(EXCEPTION) << "can not find group for given group name " << groupName;
    return nullptr;
  }
  return communicationGroups[groupName];
}

uint32_t CollectiveManager::GetGroupRank(const std::string& groupName) {
  if (communicationGroups.find(groupName) == communicationGroups.end()) {
    RT_GLOG(ERROR) << "can not find group for given group name " << groupName;
    return false;
  }
  return communicationGroups[groupName]->group_rank();
}

uint32_t CollectiveManager::GetGroupSize(const std::string& groupName) {
  if (communicationGroups.find(groupName) == communicationGroups.end()) {
    RT_GLOG(ERROR) << "can not find group for given group name " << groupName;
    return false;
  }
  return communicationGroups[groupName]->group_size();
}

void CollectiveManager::SetGlobalRankId(uint32_t globalRankId) {
  this->globalRankId = globalRankId;
}

void CollectiveManager::SetGlobalRankSize(uint32_t globalRankSize) {
  this->globalRankSize = globalRankSize;
}

void CollectiveManager::SetLocalRankId(uint32_t localRankId) {
  this->localRankId = localRankId;
}

uint32_t CollectiveManager::global_rank_id() const {
  return globalRankId;
}

uint32_t CollectiveManager::local_rank_id() const {
  return localRankId;
}

uint32_t CollectiveManager::global_rank_size() const {
  return globalRankSize;
}

} // namespace collective
} // namespace fxrt
