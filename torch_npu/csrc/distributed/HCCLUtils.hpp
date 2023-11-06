#pragma once

#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/NPUErrorCodes.h"
#include <memory>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define HCCL_CHECK_ERROR(cmd)                                           \
    do {                                                                \
        HcclResult error = cmd;                                         \
        if (error != HCCL_SUCCESS) {                                    \
            std::string err = "[ERROR] HCCL error in: " +               \
                std::string(__FILE__) +                                 \
                ":" + std::to_string(__LINE__) + ".\n" +                \
                c10_npu::acl::AclGetErrMsg();                           \
            throw std::runtime_error(err);                              \
        }                                                               \
    } while (0)

#define ENABLE_HCCL_ERROR_CHECKING

namespace c10d_npu {

// RAII wrapper for HCCL communicator
class HCCLComm {
public:
    explicit HCCLComm(HcclComm hcclComm) : hcclComm_(hcclComm) {}

    HCCLComm() : HCCLComm(nullptr) {}

    ~HCCLComm() {
        destropyHcclComm();
    }

    static std::shared_ptr<HCCLComm> create(
        int numRanks,
        int rank,
        HcclRootInfo& rootInfo) {
        auto comm = std::make_shared<HCCLComm>();
        HCCL_CHECK_ERROR(HcclCommInitRootInfo(numRanks, &rootInfo, rank, &(comm->hcclComm_)));
        c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn([=]() ->void {comm->destropyHcclComm();},
                                                             c10_npu::ReleasePriority::PriorityMiddle);
        return comm;
    }

    // Must not be copyable
    HCCLComm(const HCCLComm&) = delete;
    HCCLComm& operator=(const HCCLComm&) = delete;

    // Move constructable
    HCCLComm(HCCLComm&& other) {
        std::swap(hcclComm_, other.hcclComm_);
    }

    // Move assignable
    HCCLComm& operator=(HCCLComm&& other) {
        std::swap(hcclComm_, other.hcclComm_);
        return *this;
    }

    HcclComm getHcclComm() const{
        return hcclComm_;
    }

    void destropyHcclComm() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (hcclComm_) {
            HcclCommDestroy(hcclComm_);
            hcclComm_ = nullptr;
        }
    }

protected:
    HcclComm hcclComm_;
    mutable std::mutex mutex_;
};
} // namespace c10d_npu
