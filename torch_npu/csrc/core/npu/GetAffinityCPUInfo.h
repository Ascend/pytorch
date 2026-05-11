#ifndef THNP_GETAFFINITY_INC
#define THNP_GETAFFINITY_INC
#include <set>

namespace c10_npu {
using CoreId = unsigned int;
using CoreIdList = std::set<CoreId>;

CoreIdList GetAffinityCores(int card_id);
} // namespace c10_npu
#endif