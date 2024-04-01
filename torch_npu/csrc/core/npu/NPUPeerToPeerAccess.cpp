#include <vector>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <third_party/acl/inc/acl/acl_rt.h>
#include "torch_npu/csrc/core/npu/NPUPeerToPeerAccess.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace at_npu {
namespace native {

NpuP2pCtrl::NpuP2pCtrl()
{
    num_devices_ = c10_npu::device_count();

    device_enabled_count_.clear();
    device_enabled_count_.resize(num_devices_, 1);

    p2p_access_enabled_cache_.clear();
    p2p_access_enabled_cache_.resize(num_devices_ * num_devices_, P2pStatus::UNKONWN);

    for (const auto i : c10::irange(num_devices_)) {
        p2p_access_enabled_cache_[i * num_devices_ + i] = P2pStatus::COPY_ALLOWED;
    }
}

// Get NpuSysCtrl singleton instance
NpuP2pCtrl& NpuP2pCtrl::get_instance()
{
    static NpuP2pCtrl instance;
    return instance;
}

void NpuP2pCtrl::enable_peer_access(int32_t source_dev, int32_t dest_dev)
{
    uint32_t enable_flag_value = 0;
    c10_npu::NPUGuard guard(source_dev);
    NPU_CHECK_ERROR(aclrtDeviceEnablePeerAccess(dest_dev, enable_flag_value));
    return;
}

// Check whether the two devices are enabled by p2p and tensor can be copied
bool NpuP2pCtrl::get_p2p_access(int32_t source_dev, int32_t dest_dev, bool& flag)
{
    TORCH_INTERNAL_ASSERT(num_devices_ >= 0, "p2p access cache not initialized");
    TORCH_CHECK(source_dev >= 0 && source_dev < num_devices_, source_dev, " is not a device", PTA_ERROR(ErrCode::VALUE));
    TORCH_CHECK(dest_dev >= 0 && dest_dev < num_devices_, dest_dev, " is not a device", PTA_ERROR(ErrCode::VALUE));

    // get access source_dev -> dest_dev
    auto &cache_s2d = p2p_access_enabled_cache_[source_dev * num_devices_ + dest_dev];
    auto &cache_d2s = p2p_access_enabled_cache_[dest_dev * num_devices_ + source_dev];
    
    if (cache_s2d != P2pStatus::UNKONWN) {
        return static_cast<bool>(cache_s2d);
    }

    if (device_enabled_count_[source_dev] >= C10_P2P_ACCESS_MAX_NPUS ||
        device_enabled_count_[dest_dev] >= C10_P2P_ACCESS_MAX_NPUS) {
        // we enable P2P in groups of 8.
        cache_s2d = P2pStatus::COPY_NOT_ALLOWED;
        cache_d2s = P2pStatus::COPY_NOT_ALLOWED;
        std::string warning_str {};
        for (auto i = 0; i < num_devices_; i++) {
            if (p2p_access_enabled_cache_[source_dev * num_devices_ + i] == P2pStatus::COPY_ALLOWED) {
                warning_str += std::to_string(i);
                warning_str += ", ";
            }
        }
        ASCEND_LOGW("The NPU device is %d, and try to copy and enable p2p with %d. ", source_dev, dest_dev);
        ASCEND_LOGW(
            "However the max number of npus in P2P group is 8. "
            "Currently NPU device %d has already enable with 8 device, they are %s", source_dev, warning_str.c_str());
        return static_cast<bool>(cache_s2d);
    }
    // The aclrtEnablePeerAccess capability is not equal to cuda,
    // And both sides need to be enabled to activate the p2p capability
    int32_t result_s2d = -1;
    int32_t result_d2s = -1;
    NPU_CHECK_ERROR(aclrtDeviceCanAccessPeer(&result_s2d, source_dev, dest_dev));
    NPU_CHECK_ERROR(aclrtDeviceCanAccessPeer(&result_d2s, dest_dev, source_dev));
    // A two-way enable is required
    if (!result_s2d || !result_d2s) {
        cache_s2d = P2pStatus::COPY_NOT_ALLOWED;
        cache_d2s = P2pStatus::COPY_NOT_ALLOWED;
        flag = true;
        return static_cast<bool>(cache_s2d);
    }

    enable_peer_access(source_dev, dest_dev);
    enable_peer_access(dest_dev, source_dev);
    // A two-way enable is required
    cache_s2d = P2pStatus::COPY_ALLOWED;
    cache_d2s = P2pStatus::COPY_ALLOWED;
    // Update p2p enable count
    device_enabled_count_[source_dev]++;
    device_enabled_count_[dest_dev]++;

    return static_cast<bool>(cache_s2d);
}

} // namespace native
} // namespace at_npu
