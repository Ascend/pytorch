#pragma once

#include <atomic>
#include <mutex>
#include <map>

#include "third_party/acl/inc/acl/acl_prof.h"

#include "torch_npu/csrc/toolkit/profiler/common/singleton.h"
#include "torch_npu/csrc/toolkit/profiler/inc/data_dumper.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
namespace torch_npu {
namespace profiler {
constexpr uint64_t Level_none = 0;
constexpr uint64_t Level0 = ACL_PROF_TASK_TIME_L0 | ACL_PROF_ACL_API;
constexpr uint64_t Level1 = ACL_PROF_TASK_TIME | ACL_PROF_ACL_API | ACL_PROF_HCCL_TRACE | ACL_PROF_AICORE_METRICS;
constexpr uint64_t Level2 = Level1 | ACL_PROF_RUNTIME_API | ACL_PROF_AICPU;

struct NpuTraceConfig {
    std::string trace_level;
    std::string metrics;
    bool npu_memory;
    bool l2_cache;
    bool record_op_args;
    bool msprof_tx;
    bool op_attr;
    std::vector<std::string> host_sys;
    std::vector<std::string> mstx_domain_include;
    std::vector<std::string> mstx_domain_exclude;
    bool sys_io;
    bool sys_interconnection;
};

C10_NPU_API int8_t GetTraceLevel();

class ProfilerMgr {
public:
    void Init(const std::string &path, bool npu_trace);
    void Warmup(const NpuTraceConfig &npu_config, bool cpu_trace);
    void Start(const NpuTraceConfig &npu_config, bool cpu_trace);
    void Stop();
    void Finalize();
    void Upload(std::unique_ptr<torch_npu::toolkit::profiler::BaseReportData> data);
    void UploadWithLock(std::unique_ptr<torch_npu::toolkit::profiler::BaseReportData> data);
    void UploadTraceEventData(std::unique_ptr<torch_npu::toolkit::profiler::PythonTracerFuncData> data);
    void UploadTraceHashData(std::unique_ptr<torch_npu::toolkit::profiler::PythonTracerHashData> data);
    void UploadParamData(std::unique_ptr<torch_npu::toolkit::profiler::ParamTensorData> data);
    int8_t GetTraceLevel();
    static ProfilerMgr *GetInstance();
    std::atomic<bool>& GetNpuTrace()
    {
        return npu_trace_;
    }

    std::atomic<bool>& GetMsprofTx()
    {
        return msprof_tx_;
    }

    std::atomic<bool>& ReportEnable()
    {
        return report_enable_;
    }

    std::atomic<bool>& ReportMemEnable()
    {
        return profile_memory_;
    }

    bool IsMstxDomainEnabled(const std::string &domainName);

private:
    ProfilerMgr();
    explicit ProfilerMgr(const ProfilerMgr &obj) = delete;
    ProfilerMgr& operator=(const ProfilerMgr &obj) = delete;
    explicit ProfilerMgr(ProfilerMgr &&obj) = delete;
    ProfilerMgr& operator=(ProfilerMgr &&obj) = delete;
    void EnableMsProfiler(uint32_t *deviceIdList, uint32_t deviceNum, aclprofAicoreMetrics aicMetrics, uint64_t dataTypeConfig);
    void WarmupMsProfiler(uint32_t *deviceIdList, uint32_t deviceNum, aclprofAicoreMetrics aicMetrics, uint64_t dataTypeConfig);
    uint64_t PrepareProfilerConfig(const NpuTraceConfig &npu_config);
    void PrepareProfilerDeviceSysConfig(const NpuTraceConfig &npu_config);
    void PrepareProfilerHostSysConfig(const std::vector<std::string> &host_sys);
    aclprofAicoreMetrics PrepareProfilerAicMetrics(const NpuTraceConfig &npu_config);
    uint64_t CheckFeatureConfig(uint64_t datatype_config);
    void StartDataReceiver(const std::string &fwk_path);
    void StopDataReceiver();

private:
    static std::map<std::string, aclprofAicoreMetrics> npu_metrics_map_;
    static std::map<std::string, uint64_t> trace_level_map_;
    std::atomic<bool> report_enable_;
    std::atomic<bool> npu_trace_;
    std::atomic<bool> record_op_args_;
    std::atomic<bool> profile_memory_;
    std::atomic<bool> msprof_tx_;
    std::vector<std::string> mstx_domain_include_;
    std::vector<std::string> mstx_domain_exclude_;
    std::atomic<bool> enable_warmup_;
    std::atomic<int8_t> trace_level_;
    std::string path_;
    aclprofConfig *profConfig_;
    torch_npu::toolkit::profiler::DataDumper dataReceiver_;
    torch_npu::toolkit::profiler::TraceDataDumper traceDataReceiver_;
    std::mutex reportDataMutex_;
    torch_npu::toolkit::profiler::DataDumper dataReceiverWithLock_;
};
} // profiler
} // torch_npu
