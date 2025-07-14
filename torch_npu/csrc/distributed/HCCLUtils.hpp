#pragma once
#include <map>
#include <memory>
#include <string>

#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include "third_party/hccl/inc/hccl/hccl.h"
#include "third_party/hccl/inc/hccl/hccl_types.h"

#define HCCL_CHECK_ERROR(err_code, ...)                                      \
    do {                                                                     \
        auto Error = err_code;                                               \
        if ((Error) != HCCL_SUCCESS) {                                       \
            CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE(Error);              \
            if (c10_npu::option::OptionsManager::IsCompactErrorOutput()) {   \
                std::ostringstream oss;                                      \
                oss << " HCCL function error: " << getErrorFunction(#err_code, ##__VA_ARGS__)    \
                   << ", error code is " << Error << " "                    \
                   << DIST_ERROR(ErrCode::HCCL) + ".\n";                     \
                std::string err_msg = oss.str();                          \
                ASCEND_LOGE("%s", err_msg.c_str());                       \
                TORCH_CHECK(                                                 \
                    false,                                                   \
                    c10_npu::c10_npu_get_error_message());                   \
            } else {                                                         \
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
                c10_npu::c10_npu_get_error_message());                               \
        }                                                                    \
    }                                                                       \
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
            TORCH_CHECK(false, err, DIST_ERROR(ErrCode::HCCL));                     \
        }                                                                           \
    } while (0)

namespace c10d_npu {
extern HcclResult hcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError);
extern HcclResult hcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclCommConfig* config, HcclComm *comm);
extern HcclResult hcclCommInitClusterInfoConfig(const char *clusterInfo, uint32_t rank, HcclCommConfig *config, HcclComm *comm);
extern HcclResult hcclCreateSubCommConfig(HcclComm *comm, uint32_t rankNum, uint32_t *rankIds, uint64_t subCommId, uint32_t subCommRankId,
    HcclCommConfig* config, HcclComm *subComm);
extern HcclResult hcclCommWorkingDevNicSet(HcclComm comm, uint32_t *ranks, bool *useBackup, uint32_t nRanks);

// Provides additional detail into HCCL error codes based on when these are
// thrown in the HCCL codebase.
std::string getHcclErrorDetailStr(
    HcclResult error,
    c10::optional<std::string> processGroupFailureReason = c10::nullopt);

HcclDataType getHcclDataType(at::ScalarType type);

std::string getHcclDataTypeSerialString(HcclDataType type);

bool isFileExists(const std::string& path);

bool checkFilePathReadable(const std::string& file);

bool isSupportHcclCommName();

// RAII wrapper for HCCL communicator
class C10_NPU_API HCCLComm {
public:
    explicit HCCLComm(HcclComm hcclComm);
    HCCLComm() : HCCLComm(nullptr) {}
    ~HCCLComm();

    static std::shared_ptr<HCCLComm> create(
        int numRanks,
        int rank,
        HcclRootInfo& rootInfo);

    static std::shared_ptr<HCCLComm> create_config(
        int numRanks,
        int rank,
        HcclRootInfo& rootInfo,
        HcclCommConfig* config);

    static std::shared_ptr<HCCLComm> createGlobalHcclComm(
        const char *clusterInfo,
        uint32_t rank,
        HcclCommConfig* config);

    static std::shared_ptr<HCCLComm> createSubHcclComm(
        std::shared_ptr<HCCLComm> comm,
        uint32_t rankNum,
        uint32_t *rankIds,
        uint64_t subCommId,
        uint32_t subCommRankId,
        HcclCommConfig* config);

    int hcclCommType;
    int p2pPeer;

    // Must not be copyable
    HCCLComm(const HCCLComm&) = delete;
    HCCLComm& operator=(const HCCLComm&) = delete;

    // Move constructable
    HCCLComm(HCCLComm&& other);

    // Move assignable
    HCCLComm& operator=(HCCLComm&& other);

    HcclComm getHcclComm() const
    {
        return hcclComm_;
    }

    void destroyHcclComm();

    HcclResult checkForHcclError();

protected:
    HcclComm hcclComm_;
    mutable std::mutex mutex_;
    HcclResult hcclAsyncErr_;
};

class TORCH_API DebugInfoWriter {
public:
    virtual ~DebugInfoWriter() = default;
    virtual void write(const std::string &hcclTrace);
    static DebugInfoWriter &getWriter(int rank);
    static void registerWriter(std::unique_ptr<DebugInfoWriter> writer);
    virtual std::string getWriterTarget()
    {
        return filename_;
    }

protected:
    DebugInfoWriter(std::string namePrefix, int rank)
    {
        filename_ = c10::str(namePrefix, rank);
    }
    std::string filename_;

private:
    static std::unique_ptr<DebugInfoWriter> writer_;
    static std::atomic<bool> hasWriterRegistered_;
};
} // namespace c10d_npu
