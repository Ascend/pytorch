#include "hardware/hardware_abstract/collective/communication_group.h"

namespace fxrt {
namespace collective {
CommunicationGroup::CommunicationGroup(
    const std::string& name,
    const std::vector<uint32_t>& groupRanks,
    uint32_t groupRank,
    int64_t comm)
    : groupName(name), groupRanks(groupRanks), groupRank(groupRank), comm(comm) {}

const std::string& CommunicationGroup::group_name() const {
  return groupName;
}

const std::vector<uint32_t>& CommunicationGroup::group_ranks() const {
  return groupRanks;
}

uint32_t CommunicationGroup::group_size() const {
  return groupRanks.size();
}

uint32_t CommunicationGroup::group_rank() const {
  return groupRank;
}

int64_t CommunicationGroup::communicator() const {
  return comm;
}

} // namespace collective
} // namespace fxrt
