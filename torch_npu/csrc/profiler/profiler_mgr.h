#ifndef TORCH_NPU_TOOLKIT_PROFILER_PROFILER_MGR_INC
#define TORCH_NPU_TOOLKIT_PROFILER_PROFILER_MGR_INC
#include <atomic>
#include <map>

#include "third_party/acl/inc/acl/acl_prof.h"

#include "torch_npu/csrc/toolkit/profiler/common/singleton.h"
#include "torch_npu/csrc/toolkit/profiler/inc/data_dumper.h"
namespace torch_npu {
namespace profiler {

enum class ProfLevel {
  MSPROF_TRACE_NONE,
  MSPROF_TRACE_NPU,
};

class ProfilerMgr : public torch_npu::toolkit::profiler::Singleton<ProfilerMgr> {
friend class torch_npu::toolkit::profiler::Singleton<ProfilerMgr>;
public:
  void Init(const std::string &path, bool npu_trace);
  void Start(ProfLevel level, bool cpu_trace);
  void Stop();
  void Finalize();
  void Upload(std::unique_ptr<torch_npu::toolkit::profiler::BaseReportData> data);
  bool NpuProfEnable() {
    return npu_trace_.load();
  }

  bool ReportEnable() {
    return report_enable_.load();
  }

private:
  ProfilerMgr();
  explicit ProfilerMgr(const ProfilerMgr &obj) = delete;
  ProfilerMgr& operator=(const ProfilerMgr &obj) = delete;
  explicit ProfilerMgr(ProfilerMgr &&obj) = delete;
  ProfilerMgr& operator=(ProfilerMgr &&obj) = delete;
  void EnableMsProfiler(uint32_t *deviceIdList, uint32_t deviceNum, aclprofAicoreMetrics aicMetrics, uint64_t dataTypeConfig);

private:
  static std::map<int32_t, uint64_t> dataTypeConfigMap_;
  std::atomic<bool> report_enable_;
  std::atomic<bool> npu_trace_;
  std::string path_;
  aclprofConfig *profConfig_;
  torch_npu::toolkit::profiler::DataDumper dataReceiver_;
};
} // profiler
} // torch_npu
#endif
