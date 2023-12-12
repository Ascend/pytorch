// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/NPUErrorCodes.h"
#include <memory>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#define HCCL_CHECK_ERROR(cmd)                                       \
    do {                                                              \
        HcclResult error = cmd;                                         \
        if (error != HCCL_SUCCESS) {                                    \
            std::string err = "[ERROR] HCCL error in: " +                 \
                std::string(__FILE__) +                                   \
                ":" + std::to_string(__LINE__) + ".\n" +                  \
                c10_npu::acl::AclGetErrMsg();                             \
            throw std::runtime_error(err);                                \
        }                                                               \
    } while (0)

#define ENABLE_HCCL_ERROR_CHECKING

// Macro to throw on a non-successful HCCL return value.
#define C10D_HCCL_CHECK(cmd)                                                  \
    do {                                                                        \
        HcclResult result = cmd;                                                \
        if (result != HCCL_SUCCESS) {                                              \
            std::string err = "HCCL error in: " + std::string(__FILE__) + ":" +     \
                std::to_string(__LINE__) + ", " +                                   \
                "\n" + getHcclErrorDetailStr(result);                \
            TORCH_CHECK(false, err);                                  \
        }                                                                         \
    } while (0)

namespace c10d_npu {
extern HcclResult hcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError);

// Provides additional detail into HCCL error codes based on when these are
// thrown in the HCCL codebase.
std::string getHcclErrorDetailStr(
    HcclResult error,
    c10::optional<std::string> processGroupFailureReason = c10::nullopt);

// RAII wrapper for HCCL communicator
class HCCLComm {
public:
  explicit HCCLComm(HcclComm hcclComm) :
      hcclComm_(hcclComm),
      hcclAsyncErr_(HCCL_SUCCESS) {}

  HCCLComm() : HCCLComm(nullptr) {}

  ~HCCLComm() {
    destropyHcclComm();
  }

    static std::shared_ptr<HCCLComm> create(
        int numRanks,
        int rank,
        HcclRootInfo& rootInfo) {
        auto comm = std::make_shared<HCCLComm>();
        HCCL_CHECK_ERROR(
            HcclCommInitRootInfo(numRanks, &rootInfo, rank, &(comm->hcclComm_)));
        c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn([=]() ->void {
            comm->destropyHcclComm();
            }, c10_npu::ReleasePriority::PriorityMiddle);

        comm->hcclId_ = rootInfo;
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

    HcclRootInfo getHcclId() {
        return hcclId_;
    }

    HcclResult checkForHcclError() {
      std::unique_lock<std::mutex> lock(mutex_);
#ifdef ENABLE_HCCL_ERROR_CHECKING
        if (hcclAsyncErr_ != HCCL_SUCCESS) {
          return hcclAsyncErr_;
        }
        if (hcclComm_ != nullptr) {
            auto ret = hcclGetCommAsyncError(hcclComm_, &hcclAsyncErr_);
            if (ret != HCCL_SUCCESS) {
                TORCH_NPU_WARN("hcclGetCommAsyncError interface query error");
                ASCEND_LOGD("hcclGetCommAsyncError interface query error");
                return HCCL_SUCCESS;
            }
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
    HcclRootInfo hcclId_ = {""};
    c10::optional<std::string> commFailureReason_;
    HcclResult hcclAsyncErr_;
};
} // namespace c10d_npu
