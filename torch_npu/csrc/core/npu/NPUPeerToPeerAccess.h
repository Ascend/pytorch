#pragma once

#include <cstdint>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace at_npu {
namespace native {

class NpuP2pCtrl {
public:
    // Values include "1" (copy allowed), "0" (copy not allowed), and "-1" (unknown).
    enum class P2pStatus : int8_t {
        UNKONWN = -1,
        COPY_NOT_ALLOWED = 0,
        COPY_ALLOWED = 1
    };

    ~NpuP2pCtrl() = default;

    // Get NpuP2pCtrl singleton instance
    static NpuP2pCtrl& get_instance();

    // Check whether the two devices are enabled by p2p and tensor can be copied
    bool get_p2p_access(int32_t source_dev, int32_t dest_dev, bool& flag);

private:
    NpuP2pCtrl();

    void enable_peer_access(int32_t source_dev, int32_t dest_dev);

    // Use a 1-dimensional vector to store 2-dimensional data
    std::vector<P2pStatus> p2p_access_enabled_cache_;
    // record each device p2p enable count.
    // p2pAccessEnabled records if p2p copies are allowed between pairs of devices.
    // Currently the max number of npus in P2P group is 8, so if there are more
    std::vector<int8_t> device_enabled_count_;
    // init flag
    int64_t num_devices_;
};

} // namespace native
} // namespace at_npu
