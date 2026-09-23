#ifndef FXRT_SRC_HARDWARE_COLLECTIVE_COMMUNICATION_GROUP_H_
#define FXRT_SRC_HARDWARE_COLLECTIVE_COMMUNICATION_GROUP_H_

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "common/visible.h"

namespace fxrt {
namespace collective {

class DA_API CommunicationGroup {
 public:
  explicit CommunicationGroup(
      const std::string& name,
      const std::vector<uint32_t>& groupRanks,
      uint32_t groupRank,
      int64_t comm);

  ~CommunicationGroup() = default;

  virtual const std::string& group_name() const;
  virtual const std::vector<uint32_t>& group_ranks() const;
  virtual uint32_t group_rank() const;
  virtual uint32_t group_size() const;
  virtual int64_t communicator() const;

 protected:
  std::string groupName;
  std::vector<uint32_t> groupRanks;
  uint32_t groupRank;
  int64_t comm;
};

using CommunicationGroupPtr = std::shared_ptr<CommunicationGroup>;
} // namespace collective
} // namespace fxrt

#endif // FXRT_SRC_HARDWARE_COLLECTIVE_COMMUNICATION_GROUP_H_
