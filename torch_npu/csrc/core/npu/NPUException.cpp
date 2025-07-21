#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
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
    if (c10_npu::option::OptionsManager::IsCompactErrorOutput()) {
        return "";
    }
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

std::unordered_map<int, std::function<std::string(int)>> errCodeHandlerMap = {
    {ACL_ERROR_RT_DEVICE_TASK_ABORT, std::bind(&handleDeviceTaskAbort, std::placeholders::_1)},
    {ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR, std::bind(&handleHbmMultiBitEccError, std::placeholders::_1)},
    {ACL_ERROR_RT_DEVICE_MEM_ERROR, std::bind(&handleDeviceMemError, std::placeholders::_1)},
    {ACL_ERROR_RT_SUSPECT_DEVICE_MEM_ERROR, std::bind(&handleSuspectDeviceMemError, std::placeholders::_1)},
    {ACL_ERROR_RT_LINK_ERROR, std::bind(&handleLinkError, std::placeholders::_1)},
    {ACL_ERROR_RT_COMM_OP_RETRY_FAIL, std::bind(&handleHcclOpRetryFailed, std::placeholders::_1)}
};

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

const std::string c10_npu_check_error_message(std::string& errmsg)
{
    static const std::regex errorRegex(R"(^E[1-9A-Z]9999)");
    if (std::regex_search(errmsg, errorRegex)) {
        return "CANN Inner Error. Please rectify the fault based on the error information in the ascend log.";
    }

    std::regex dateRegex(R"(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}\.\d{3}\.\d{3})");
    std::smatch match;

    if (std::regex_search(errmsg, match, dateRegex)) {
        size_t dateEndPos = match.position(0) + match.length(0);
        size_t tracePos = errmsg.find("TraceBack (most recent call last):\n", dateEndPos);
        std::string content;
        if (tracePos != std::string::npos) {
            content = errmsg.substr(dateEndPos, tracePos - dateEndPos);
        } else {
            content = errmsg.substr(dateEndPos);
        }

        std::regex ws_regex("[\\s\\t\\n\\r]+");
        content = std::regex_replace(content, ws_regex, " ");
        if (!content.empty() && content.front() == ' ') {
            content.erase(0, 1);
        }
        if (!content.empty() && content.back() == ' ') {
            content.pop_back();
        }

        return content;
    }

    return "";
}


const char *c10_npu_get_error_message()
{
    auto errmsg = c10_npu::acl::AclGetErrMsg();
    if (c10_npu::option::OptionsManager::IsCompactErrorOutput()) {
        std::string log(errmsg);
        std::string errmsg_ = c10_npu::c10_npu_check_error_message(log);
        thread_local std::string processedErrMsg = "CANN error: " + errmsg_;
        c10_npu::setRepoErrMsg(processedErrMsg.c_str());
        return processedErrMsg.c_str();
    } else {
        c10_npu::setRepoErrMsg(errmsg);
        return errmsg;
    }
}

void record_mem_hbm_ecc_error()
{
    MemUceInfo memUceInfo_;
    memUceInfo_.is_hbm_ecc_error = true;
    ASCEND_LOGE("Log HBM MULTI BIT ECC ERROR, set is_hbm_ecc_error param is true");
    set_mem_uce_info(memUceInfo_);
}

bool checkUceErrAndRepair(bool check_error, std::string& err_msg)
{
    int device = 0;
    auto err = c10_npu::GetDevice(&device);
    if (err != ACL_ERROR_NONE) {
        err_msg = "ERROR happened in GetDevice.";
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

std::string handleDeviceTaskAbort(int errorCode)
{
    ASCEND_LOGE("getRepoStopFlag in Run, throw FORCE STOP.");
    return "FORCE STOP";
}

std::string handleHbmMultiBitEccError(int errorCode)
{
    ASCEND_LOGE("getRepoStopFlag in Run, throw ECC ERROR.");
    std::string error_msg(c10_npu::c10_npu_get_error_message());
    std::regex pattern(R"(time us= (\d+)\.)");
    std::smatch match;
    std::string time_msg = "";
    if (std::regex_search(error_msg, match, pattern)) {
        if (match.size() > 1) {
            time_msg = match[1].str();
        }
    }
    c10_npu::record_mem_hbm_ecc_error();
    return "HBM MULTI BIT ECC ERROR." + error_msg + "time is " + time_msg;
}

std::string handleDeviceMemError(int errorCode)
{
    std::string error_msg = "";
    if (c10_npu::checkUceErrAndRepair(true, error_msg)) {
        ASCEND_LOGE("getRepoStopFlag in Run, throw UCE ERROR.");
        return "UCE ERROR";
    }
    return "";
}

std::string handleSuspectDeviceMemError(int errorCode)
{
    ASCEND_LOGE("getRepoStopFlag in Run, throw SUSPECT MEM ERROR.");
    return "SUSPECT MEM ERROR";
}

std::string handleLinkError(int errorCode)
{
    ASCEND_LOGE("getRepoStopFlag in Run, throw HCCS LINK ERROR.");
    return "HCCS LINK ERROR";
}

std::string handleHcclOpRetryFailed(int errorCode)
{
    ASCEND_LOGE("getRepoStopFlag in Run, throw HCCL OP RETRY FAILED.");
    return "HCCL OP RETRY FAILED";
}

std::string handleDeviceError(int errorCode)
{
    auto handlerIter = errCodeHandlerMap.find(errorCode);
    if (handlerIter != errCodeHandlerMap.end()) {
        std::function<std::string(int)> handler = handlerIter->second;
        if (handler != nullptr) {
            return handler(errorCode);
        }
    }
    return "";
}

} // namespace c10_npu
