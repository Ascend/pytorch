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

bool has_throw_error = false;

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
    memUceInfo.device = 0;
    memUceInfo.info.clear();
    memUceInfo.retSize = 0;
}

const char *c10_npu_get_error_message()
{
    return c10_npu::acl::AclGetErrMsg();
}

bool checkUceErrAndRepair()
{
    int device = 0;
    auto err = c10_npu::GetDevice(&device);
    if (err != ACL_ERROR_NONE) {
        TORCH_CHECK(false, "ERROR happend in GetDevice.", PTA_ERROR(ErrCode::ACL))
    }

    aclrtMemUceInfo info[MAX_MEM_UCE_INFO_ARRAY_SIZE];
    size_t retSize = 0;

    err = c10_npu::acl::AclrtGetMemUceInfo(device, info, sizeof(info) / sizeof(aclrtMemUceInfo), &retSize);
    if (err == ACL_ERROR_NONE) {
        if (retSize > 0) {
            ASCEND_LOGE("AclrtGetMemUceInfo get UCE ERROR, retSize is %d", retSize);
            MemUceInfo memUceInfo_;
            memUceInfo_.device = device;
            memUceInfo_.info.assign(info, info + retSize);
            memUceInfo_.retSize = retSize;
            set_mem_uce_info(memUceInfo_);
            NPU_CHECK_ERROR_WITHOUT_UCE(c10_npu::acl::AclrtMemUceRepair(device, info, retSize));
            return true;
        } else {
            return false;
        }
    } else {
        static c10_npu::acl::AclErrorCode err_map;
        TORCH_CHECK(false, __func__, ":", __FILE__, ":", __LINE__, " NPU error, error code is ", err, PTA_ERROR(ErrCode::ACL),
                    (err_map.error_code_map.find(err) != err_map.error_code_map.end() ?
                    "\n[Error]: " + err_map.error_code_map[err] : "."), "\n", c10_npu::c10_npu_get_error_message());
    }

    return false;
}

bool get_has_throw_error()
{
    return has_throw_error;
}

void set_has_throw_error(bool flag)
{
    has_throw_error = flag;
}

} // namespace c10_npu
