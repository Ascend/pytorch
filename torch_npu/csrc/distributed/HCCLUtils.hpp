#pragma once

#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include <map>
#include <memory>
#include <string>
#include <fstream>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define HCCL_CHECK_ERROR(err_code, ...)                                      \
    do {                                                                     \
        auto Error = err_code;                                               \
        if ((Error) != HCCL_SUCCESS) {                                       \
            TORCH_CHECK(                                                     \
                false,                                                       \
                __func__,                                                    \
                ":",                                                         \
                __FILE__,                                                    \
                ":",                                                         \
                __LINE__,                                                    \
                " HCCL function error: ", getErrorFunction(#err_code, ##__VA_ARGS__),   \
                ", error code is ", Error,                                   \
                DIST_ERROR(ErrCode::HCCL) + ".\n" +                          \
                c10_npu::acl::AclGetErrMsg());                               \
        }                                                                    \
    } while (0)

#define ENABLE_HCCL_ERROR_CHECKING

// Macro to throw on a non-successful HCCL return value.
#define C10D_HCCL_CHECK(cmd)                                                        \
    do {                                                                            \
        HcclResult result = cmd;                                                    \
        if (result != HCCL_SUCCESS) {                                               \
            std::string err = "HCCL error in: " + std::string(__FILE__) + ":" +     \
            std::to_string(__LINE__) + ", " +                                       \
            "\n" + getHcclErrorDetailStr(result);                                   \
            TORCH_CHECK(false, err);                                                \
        }                                                                           \
    } while (0)

namespace c10d_npu {
extern HcclResult hcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError);
extern HcclResult hcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclCommConfig* config, HcclComm *comm);

// Provides additional detail into HCCL error codes based on when these are
// thrown in the HCCL codebase.
std::string getHcclErrorDetailStr(
    HcclResult error,
    c10::optional<std::string> processGroupFailureReason = c10::nullopt);

HcclDataType getHcclDataType(at::ScalarType type);

std::string getHcclDataTypeSerialString(HcclDataType type);

bool isFileExists(std::string& name);

// RAII wrapper for HCCL communicator
class HCCLComm {
public:
    explicit HCCLComm(HcclComm hcclComm) : hcclComm_(hcclComm), hcclAsyncErr_(HCCL_SUCCESS) {}

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

    static std::shared_ptr<HCCLComm> create_config(
        int numRanks,
        int rank,
        HcclRootInfo& rootInfo,
        HcclCommConfig* config)
    {
        auto comm = std::make_shared<HCCLComm>();
        HCCL_CHECK_ERROR(hcclCommInitRootInfoConfig(numRanks, &rootInfo, rank, config, &(comm->hcclComm_)));
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
        std::swap(hcclAsyncErr_, other.hcclAsyncErr_);
    }

    // Move assignable
    HCCLComm& operator=(HCCLComm&& other) {
        std::swap(hcclComm_, other.hcclComm_);
        std::swap(hcclAsyncErr_, other.hcclAsyncErr_);
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

    HcclResult checkForHcclError() {
        std::unique_lock<std::mutex> lock(mutex_);
#ifdef ENABLE_HCCL_ERROR_CHECKING
        if (hcclAsyncErr_ != HCCL_SUCCESS) {
            return hcclAsyncErr_;
        }
        if (hcclComm_ != nullptr) {
            C10D_HCCL_CHECK(hcclGetCommAsyncError(hcclComm_, &hcclAsyncErr_));
        }
        return hcclAsyncErr_;
#else
        // Always return success, if error checks are disabled.
        return HCCL_SUCCESS;
#endif
    }

protected:
    HcclComm hcclComm_;
    mutable std::mutex mutex_;
    HcclResult hcclAsyncErr_;
};
} // namespace c10d_npu
