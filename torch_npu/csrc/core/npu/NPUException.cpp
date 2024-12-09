#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"


std::unordered_map<SubModule, std::string> submoduleMap = {
    {SubModule::PTA, "PTA"},
    {SubModule::OPS, "OPS"},
    {SubModule::DIST, "DIST"},
    {SubModule::GRAPH, "GRAPH"},
    {SubModule::PROF, "PROF"}
};

std::unordered_map<ErrCode, std::string> errCodeMap = {
    {ErrCode::SUC, "success"},
    {ErrCode::PARAM, "invalid parameter"},
    {ErrCode::TYPE, "invalid type"},
    {ErrCode::VALUE, "invalid value"},
    {ErrCode::PTR, "invalid pointer"},
    {ErrCode::INTERNAL, "internal error"},
    {ErrCode::MEMORY, "memory error"},
    {ErrCode::NOT_SUPPORT, "feature not supported"},
    {ErrCode::NOT_FOUND, "resource not found"},
    {ErrCode::UNAVAIL, "resource unavailable"},
    {ErrCode::SYSCALL, "system call failed"},
    {ErrCode::TIMEOUT, "timeout error"},
    {ErrCode::PERMISSION, "permission error"},
    {ErrCode::ACL, "call acl api failed"},
    {ErrCode::HCCL, "call hccl api failed"},
    {ErrCode::GE, "call ge api failed"}
};

c10::WarningHandler* getBaseHandler_()
{
    static c10::WarningHandler warning_handler_ = c10::WarningHandler();
    return &warning_handler_;
};

void warn_(const ::c10::Warning& warning)
{
    if (!c10_npu::option::OptionsManager::ShouldPrintWarning()) {
        return;
    }
  getBaseHandler_()->process(warning);
}

std::string formatErrorCode(SubModule submodule, ErrCode errorCode)
{
    std::ostringstream oss;
    int deviceIndex = -1;
    c10_npu::GetDevice(&deviceIndex);
    auto rank_id = c10_npu::option::OptionsManager::GetRankId();
    oss << "\n[ERROR] " << getCurrentTimestamp() << " (PID:" << getpid() << ", Device:" << deviceIndex << ", RankID:" << rank_id << ") ";
    oss << "ERR" << std::setw(2) << std::setfill('0') << static_cast<int>(submodule);
    oss << std::setw(3) << std::setfill('0') << static_cast<int>(errorCode);
    oss << " " << submoduleMap[submodule] << " " << errCodeMap[errorCode];

    return oss.str();
}

static std::string getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());

    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm* timeInfo = std::localtime(&currentTime);

    auto milli_time = std::chrono::duration_cast<std::chrono::milliseconds>(micros).count() % 1000;
    auto micro_time = micros.count() % 1000;

    std::ostringstream oss;
    oss << std::put_time(timeInfo, "%Y-%m-%d-%H:%M:%S");
    return oss.str();
}

namespace c10_npu {

MemUceInfo memUceInfo;

std::mutex memUceInfoMutex;

void set_mem_uce_info(MemUceInfo info)
{
    std::lock_guard<std::mutex> lock(memUceInfoMutex);
    memUceInfo = info;
}

MemUceInfo get_mem_uce_info()
{
    std::lock_guard<std::mutex> lock(memUceInfoMutex);
    return memUceInfo;
}

void clear_mem_uce_info()
{
    std::lock_guard<std::mutex> lock(memUceInfoMutex);
    memUceInfo.clear();
}

const char *c10_npu_get_error_message()
{
    return c10_npu::acl::AclGetErrMsg();
}

bool checkUceErrAndRepair(bool check_error, std::string& err_msg)
{
    int device = 0;
    auto err = c10_npu::GetDevice(&device);
    if (err != ACL_ERROR_NONE) {
        err_msg = "ERROR happend in GetDevice.";
        if (check_error) {
            TORCH_CHECK(false, err_msg, PTA_ERROR(ErrCode::ACL));
        } else {
            err_msg += PTA_ERROR(ErrCode::ACL);
            return false;
        }
    }

    MemUceInfo memUceInfo_;
    memUceInfo_.device = device;
    err = c10_npu::acl::AclrtGetMemUceInfo(device, memUceInfo_.info, sizeof(memUceInfo_.info) / sizeof(aclrtMemUceInfo), &memUceInfo_.retSize);
    if (err == ACL_ERROR_NONE) {
        if (memUceInfo_.retSize > 0) {
            ASCEND_LOGE("AclrtGetMemUceInfo get UCE ERROR, retSize is %d", memUceInfo_.retSize);
            set_mem_uce_info(memUceInfo_);
            return true;
        } else {
            err_msg = "AclrtGetMemUceInfo get UCE ERROR, retSize is " + std::to_string(memUceInfo_.retSize);
        }
    } else {
        static c10_npu::acl::AclErrorCode err_map;
        err_msg = std::string(__func__) + ":" + __FILE__ + ":" + std::to_string(__LINE__) +
                        " NPU error, error code is " + std::to_string(err) + PTA_ERROR(ErrCode::ACL) +
                        (err_map.error_code_map.find(err) != err_map.error_code_map.end() ?
                        "\n[Error]: " + err_map.error_code_map[err] : ".") +
                        "\n" + c10_npu_get_error_message();
        if (check_error) {
            TORCH_CHECK(false, err_msg);
        }
    }
    return false;
}

} // namespace c10_npu
