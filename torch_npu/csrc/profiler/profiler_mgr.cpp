#include "torch_npu/csrc/profiler/profiler_mgr.h"
#include "torch_npu/csrc/framework/interface/AclInterface.h"
#include "torch_npu/csrc/framework/interface/MsProfilerInterface.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/toolkit/profiler/common/utils.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/profiler/feature_mgr.h"

namespace torch_npu {
namespace profiler {
std::map<std::string, int8_t> trace_level_to_int_ = {
    {"Level0", 0},
    {"Level1", 1},
    {"Level2", 2},
};

std::map<std::string, aclprofAicoreMetrics> ProfilerMgr::npu_metrics_map_ = {
    {"ACL_AICORE_PIPE_UTILIZATION", ACL_AICORE_PIPE_UTILIZATION},
    {"ACL_AICORE_ARITHMETIC_UTILIZATION", ACL_AICORE_ARITHMETIC_UTILIZATION},
    {"ACL_AICORE_MEMORY_BANDWIDTH", ACL_AICORE_MEMORY_BANDWIDTH},
    {"ACL_AICORE_L0B_AND_WIDTH", ACL_AICORE_L0B_AND_WIDTH},
    {"ACL_AICORE_RESOURCE_CONFLICT_RATIO", ACL_AICORE_RESOURCE_CONFLICT_RATIO},
    {"ACL_AICORE_MEMORY_UB", ACL_AICORE_MEMORY_UB},
    {"ACL_AICORE_L2_CACHE", ACL_AICORE_L2_CACHE},
    {"ACL_AICORE_MEMORY_ACCESS", ACL_AICORE_MEMORY_ACCESS},
    {"ACL_AICORE_NONE", ACL_AICORE_NONE},
};

std::map<std::string, uint64_t> ProfilerMgr::trace_level_map_ = {
    {"Level0", Level0},
    {"Level1", Level1},
    {"Level2", Level2},
    {"Level_none", Level_none},
};

constexpr uint32_t capacity_ = 1048576;          // 2^20, Experience value for default ringbuffer size for single data
constexpr uint32_t trace_capacity_ = 128;        // 2^7, Experience value for python trace data ringbuffer size for batch data

aclprofAicoreMetrics CheckAicMetricsFeature(aclprofAicoreMetrics aic_metrics, int8_t level)
{
    if (aic_metrics == ACL_AICORE_MEMORY_ACCESS &&
        !FeatureMgr::GetInstance()->IsSupportFeature(FeatureType::FEATURE_MEMORY_ACCESS)) {
        ASCEND_LOGW("AiCMetrics is not supported to set to MemoryAccess.");
        printf("[WARN]%s,%s:%u:AiCMetrics is not supported to set to MemoryAccess, reset to default.\n",
               __FUNCTION__, __FILENAME__, __LINE__);
        return (level >= 1 ? ACL_AICORE_PIPE_UTILIZATION : ACL_AICORE_NONE);
    }
    return aic_metrics;
}

ProfilerMgr::ProfilerMgr()
    : report_enable_(false),
      npu_trace_(false),
      record_op_args_(false),
      profile_memory_(false),
      msprof_tx_(false),
      enable_warmup_(false),
      profConfig_(nullptr) {}

ProfilerMgr *ProfilerMgr::GetInstance()
{
    static ProfilerMgr instance;
    return &instance;
}

void ProfilerMgr::Init(const std::string &path, bool npu_trace)
{
    if (npu_trace) {
        at_npu::native::AclProfilingInit(path.c_str(), path.size());
        npu_trace_.store(true);
        FeatureMgr::GetInstance()->Init();
    }
    path_ = path;
}

void ProfilerMgr::WarmupMsProfiler(uint32_t *deviceIdList, uint32_t deviceNum, aclprofAicoreMetrics aicMetrics, uint64_t dataTypeConfig)
{
    profConfig_ = at_npu::native::AclProfilingCreateConfig(deviceIdList, deviceNum, aicMetrics, nullptr, dataTypeConfig);
    if (profConfig_ == nullptr) {
        ASCEND_LOGE("Create Prof Config failed.");
        return;
    }
    auto ret = at_npu::native::AclProfilingWarmup(profConfig_);
    if (ret == ACL_ERROR_PROF_MODULES_UNSUPPORTED) {
        ASCEND_LOGW("Not support aclprofWarmup.");
    } else if (ret != ACL_ERROR_NONE) {
        ASCEND_LOGE("Profiling warmup failed.");
    }
}

void ProfilerMgr::EnableMsProfiler(uint32_t *deviceIdList, uint32_t deviceNum, aclprofAicoreMetrics aicMetrics, uint64_t dataTypeConfig)
{
    // Avoid duplicate config creation in scenarios where warmup is turned on
    if (profConfig_ == nullptr) {
        profConfig_ = at_npu::native::AclProfilingCreateConfig(deviceIdList, deviceNum, aicMetrics, nullptr, dataTypeConfig);
    }
  
    if (profConfig_ == nullptr) {
        ASCEND_LOGE("Create Prof Config failed.");
        return;
    }

    auto ret = at_npu::native::AclProfilingStart(profConfig_);
    if (ret != ACL_ERROR_NONE) {
        ASCEND_LOGE("Profiling start failed.");
        return;
    }

    ASCEND_LOGI("Try to register set device callback function.");
    ret = at_npu::native::AclProfilingRegisterDeviceCallback();
    if (ret == ACL_ERROR_PROF_MODULES_UNSUPPORTED) {
        ASCEND_LOGW("Not support set device callback function.");
    } else if (ret != ACL_SUCCESS) {
        ASCEND_LOGE("Failed to register set device callback function.");
    }
}

uint64_t ProfilerMgr::PrepareProfilerConfig(const NpuTraceConfig &npu_config)
{
    auto level_iter = trace_level_map_.find(npu_config.trace_level);
    uint64_t datatype_config = (level_iter == trace_level_map_.end()) ? Level0 : trace_level_map_[npu_config.trace_level];
    auto metrics_iter = npu_metrics_map_.find(npu_config.metrics);
    if (metrics_iter != npu_metrics_map_.end() && npu_config.metrics.compare("ACL_AICORE_NONE") != 0) {
        datatype_config |= ACL_PROF_AICORE_METRICS;
    }
    if (npu_config.l2_cache) {
        datatype_config |= ACL_PROF_L2CACHE;
    }
    if (npu_config.msprof_tx) {
        datatype_config |= ACL_PROF_MSPROFTX;
    }
    if (npu_config.npu_memory) {
        datatype_config |= ACL_PROF_TASK_MEMORY;
        const std::string freq = "50";
        auto prof_ret = at_npu::native::AclprofSetConfig(ACL_PROF_SYS_HARDWARE_MEM_FREQ, freq.c_str(), freq.size());
        if (prof_ret == ACL_ERROR_PROF_MODULES_UNSUPPORTED) {
            ASCEND_LOGW("not support to set config for sys-hardware-mem.");
        }
    }
    PrepareProfilerHostSysConfig(npu_config.host_sys);
    PrepareProfilerDeviceSysConfig(npu_config);
    if (npu_config.op_attr) {
        datatype_config |= ACL_PROF_OP_ATTR;
    }
    datatype_config = CheckFeatureConfig(datatype_config);
    return datatype_config;
}

void ProfilerMgr::PrepareProfilerHostSysConfig(const std::vector<std::string> &host_sys)
{
    if (!host_sys.empty()) {
        std::string hostSysStr;
        for (size_t i = 0; i < host_sys.size(); ++i) {
            if (i > 0) {
                hostSysStr += ",";
            }
            hostSysStr += host_sys[i];
        }
        aclError hostSysRet = at_npu::native::AclprofSetConfig(ACL_PROF_HOST_SYS, hostSysStr.c_str(), hostSysStr.size());
        if (hostSysRet != ACL_SUCCESS) {
            ASCEND_LOGE("Failed call aclprofSetConfig to ACL_PROF_HOST_SYS. error_code: %d",
                static_cast<int>(hostSysRet));
        }
    }
}

void ProfilerMgr::PrepareProfilerDeviceSysConfig(const NpuTraceConfig &npu_config)
{
    if (npu_config.sys_io) {
        const std::string sysIoFreq = "100";
        aclError sysIoRet = at_npu::native::AclprofSetConfig(ACL_PROF_SYS_IO_FREQ, sysIoFreq.c_str(), sysIoFreq.size());
        if (sysIoRet != ACL_SUCCESS) {
            ASCEND_LOGW("Failed call aclprofSetConfig to ACL_PROF_SYS_IO_FREQ. error_code : %d",
                        static_cast<int>(sysIoRet));
        }
    }
    if (npu_config.sys_interconnection) {
        const std::string sysInterconnectionFreq  = "50";
        aclError sysInterconnectionRet =
          at_npu::native::AclprofSetConfig(ACL_PROF_SYS_INTERCONNECTION_FREQ, sysInterconnectionFreq.c_str(), sysInterconnectionFreq.size());
        if (sysInterconnectionRet != ACL_SUCCESS) {
            ASCEND_LOGW("Failed call aclprofSetConfig to ACL_PROF_SYS_INTERCONNECTION_FREQ. error_code : %d",
                        static_cast<int>(sysInterconnectionRet));
        }
    }
}

aclprofAicoreMetrics ProfilerMgr::PrepareProfilerAicMetrics(const NpuTraceConfig &npu_config)
{
    aclprofAicoreMetrics aic_metrics = ACL_AICORE_NONE;
    int8_t level_int = trace_level_to_int_.find(npu_config.trace_level) != trace_level_to_int_.end() ? trace_level_to_int_[npu_config.trace_level] : -1;
    auto metrics_iter = npu_metrics_map_.find(npu_config.metrics);
    if (metrics_iter != npu_metrics_map_.end() && npu_config.metrics.compare("ACL_AICORE_NONE") != 0) {
        aic_metrics = CheckAicMetricsFeature(npu_metrics_map_[npu_config.metrics], level_int);
    }
    return aic_metrics;
}

void ProfilerMgr::Warmup(const NpuTraceConfig &npu_config, bool cpu_trace)
{
    if (npu_trace_.load()) {
        auto datatype_config = PrepareProfilerConfig(npu_config);
        auto aic_metrics = PrepareProfilerAicMetrics(npu_config);
        int8_t level_int = trace_level_to_int_.find(npu_config.trace_level) != trace_level_to_int_.end() ? trace_level_to_int_[npu_config.trace_level] : -1;
        int32_t deviceId = 0;
        auto ret = c10_npu::GetDevice(&deviceId);
        if (ret != ACL_ERROR_NONE) {
            ASCEND_LOGE("Get Device ID failed.");
            return;
        }
        const uint32_t deviceNum = 1;
        uint32_t deviceIdList[deviceNum] = {deviceId};
        WarmupMsProfiler(deviceIdList, deviceNum, aic_metrics, datatype_config);
        trace_level_.store(level_int);
    }

    if (cpu_trace) {
        std::string fwk_path = path_ + "/FRAMEWORK";
        if (Utils::CreateDir(fwk_path)) {
            StartDataReceiver(fwk_path);
            report_enable_.store(true);
            profile_memory_.store(npu_config.npu_memory);
        } else {
            ASCEND_LOGE("Profiler create FRAMEWORK directory failed: %s", fwk_path.c_str());
        }
    }

    enable_warmup_.store(true);
}

void ProfilerMgr::Start(const NpuTraceConfig &npu_config, bool cpu_trace)
{
    if (npu_trace_.load()) {
        auto datatype_config = PrepareProfilerConfig(npu_config);
        auto aic_metrics = PrepareProfilerAicMetrics(npu_config);
        int8_t level_int = trace_level_to_int_.find(npu_config.trace_level) != trace_level_to_int_.end() ? trace_level_to_int_[npu_config.trace_level] : -1;
        int32_t deviceId = 0;
        auto ret = c10_npu::GetDevice(&deviceId);
        if (ret != ACL_ERROR_NONE) {
            ASCEND_LOGE("Get Device ID failed.");
            return;
        }
        const uint32_t deviceNum = 1;
        uint32_t deviceIdList[deviceNum] = {deviceId};
        EnableMsProfiler(deviceIdList, deviceNum, aic_metrics, datatype_config);
        trace_level_.store(level_int);
    }

    if (cpu_trace && enable_warmup_.load() == false) {
        std::string fwk_path = path_ + "/FRAMEWORK";
        if (Utils::CreateDir(fwk_path)) {
            StartDataReceiver(fwk_path);
            report_enable_.store(true);
            profile_memory_.store(npu_config.npu_memory);
        } else {
            ASCEND_LOGE("Profiler create FRAMEWORK directory failed: %s", fwk_path.c_str());
        }
    }
    enable_warmup_.store(false);
    msprof_tx_.store(npu_config.msprof_tx);
    mstx_domain_include_ = npu_config.mstx_domain_include;
    mstx_domain_exclude_ = npu_config.mstx_domain_exclude;
    if (npu_config.record_op_args) {
        record_op_args_.store(true);
        const std::string op_dump_path = std::string(path_.begin(), path_.begin() + path_.find_last_not_of("/") + 1) +
                                         "_op_args";
        at_npu::native::AclopStartDumpArgs(ACL_OP_DUMP_OP_AICORE_ARGS, op_dump_path.c_str());
    }
}

void ProfilerMgr::Stop()
{
    c10_npu::npuSynchronizeDevice();
    if (report_enable_.load()) {
        StopDataReceiver();
        profile_memory_.store(false);
    }
    
    if (npu_trace_.load()) {
        at_npu::native::AclProfilingStop(profConfig_);
        auto ret = at_npu::native::AclProfilingDestroyConfig(profConfig_);
        if (ret != ACL_SUCCESS) {
            ASCEND_LOGE("AclProfDestoryConfig fail, error code: %d", ret);
        }
        profConfig_ = nullptr;
    }
    msprof_tx_.store(false);
    mstx_domain_include_.clear();
    mstx_domain_exclude_.clear();
    report_enable_.store(false);
    enable_warmup_.store(false);
    if (record_op_args_.load()) {
        at_npu::native::AclopStopDumpArgs(ACL_OP_DUMP_OP_AICORE_ARGS);
        record_op_args_.store(false);
    }
}

void ProfilerMgr::Finalize()
{
    if (npu_trace_.load()) {
        at_npu::native::AclProfilingFinalize();
    }
    npu_trace_.store(false);
}

void ProfilerMgr::StartDataReceiver(const std::string &fwk_path)
{
    dataReceiver_.Init(fwk_path, capacity_);
    dataReceiver_.Start();
    traceDataReceiver_.Init(fwk_path, trace_capacity_);
    traceDataReceiver_.Start();
    dataReceiverWithLock_.Init(fwk_path, capacity_);
    dataReceiverWithLock_.Start();
}

void ProfilerMgr::StopDataReceiver()
{
    dataReceiver_.Stop();
    dataReceiver_.UnInit();
    traceDataReceiver_.Stop();
    traceDataReceiver_.UnInit();
    dataReceiverWithLock_.Stop();
    dataReceiverWithLock_.UnInit();
}

void ProfilerMgr::Upload(std::unique_ptr<torch_npu::toolkit::profiler::BaseReportData> data)
{
    if (enable_warmup_.load()) {
        return;
    }
    dataReceiver_.Report(std::move(data));
}

void ProfilerMgr::UploadWithLock(std::unique_ptr<torch_npu::toolkit::profiler::BaseReportData> data)
{
    if (enable_warmup_.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(reportDataMutex_);
    dataReceiverWithLock_.Report(std::move(data));
}

void ProfilerMgr::UploadTraceEventData(std::unique_ptr<torch_npu::toolkit::profiler::PythonTracerFuncData> data)
{
    if (enable_warmup_.load()) {
        return;
    }
    traceDataReceiver_.Report(std::move(data));
}

void ProfilerMgr::UploadTraceHashData(std::unique_ptr<torch_npu::toolkit::profiler::PythonTracerHashData> data)
{
    if (enable_warmup_.load()) {
        return;
    }
    traceDataReceiver_.ReportHash(std::move(data));
}

void ProfilerMgr::UploadParamData(std::unique_ptr<torch_npu::toolkit::profiler::ParamTensorData> data)
{
    if (enable_warmup_.load()) {
        return;
    }
    traceDataReceiver_.ReportParam(std::move(data));
}

uint64_t ProfilerMgr::CheckFeatureConfig(uint64_t datatype_config)
{
    if (!FeatureMgr::GetInstance()->IsSupportFeature(FeatureType::FEATURE_ATTR)) {
        ASCEND_LOGW("Not support to set config for ATTR.");
        return datatype_config & ~(ACL_PROF_OP_ATTR);
    }
    return datatype_config;
}

int8_t ProfilerMgr::GetTraceLevel()
{
    if (npu_trace_.load()) {
        return trace_level_.load();
    }
    return -1;
}

int8_t GetTraceLevel()
{
    return ProfilerMgr::GetInstance()->GetTraceLevel();
}

bool ProfilerMgr::IsMstxDomainEnabled(const std::string &domainName)
{
    if (mstx_domain_include_.empty() && mstx_domain_exclude_.empty()) {
        return true;
    }
    if (!mstx_domain_include_.empty()) {
        return std::find(mstx_domain_include_.begin(), mstx_domain_include_.end(), domainName) !=
            mstx_domain_include_.end();
    }
    if (!mstx_domain_exclude_.empty()) {
        return std::find(mstx_domain_exclude_.begin(), mstx_domain_exclude_.end(), domainName) ==
            mstx_domain_exclude_.end();
    }
    // both not empty, enable all domains
    return true;
}
} // profiler
} // torch_npu
