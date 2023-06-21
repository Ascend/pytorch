#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/framework/NPUDefine.h"


namespace at_npu {
namespace native {

class OpRunner {
public:
  OpRunner& Name(const string &name) {
    opName = name;
    return *this;
  };
  using PROCESS_FUNC = std::function<void()>;
  OpRunner& Func(const PROCESS_FUNC& func) {
    this->func = func;
    return *this;
  }
  void ExportParams(at_npu::native::ExecuteBsParas &params) {
    TORCH_CHECK(sizeof(ExecuteBsParas::opType) >= opName.length() + 1, "Too long IR Name: ", opName);
    memset(params.opType, '\0', sizeof(params.opType));
    opName.copy(params.opType, opName.length() + 1);
    params.paras.func = this->func;
  }

  void Run() {
    RECORD_FUNCTION(opName, std::vector<c10::IValue>({}));
    if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
#ifndef BUILD_LIBTORCH
      at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(0, op_name);
#endif
      at_npu::native::ExecuteBsParas execParams;
      ExportParams(execParams);
      c10_npu::queue::QueueParas params(c10_npu::queue::LAMBDA_EXECUTE,
                                        sizeof(at_npu::native::ExecuteBsParas),
                                        &execParams);
      c10_npu::enCurrentNPUStream(&params);
#ifndef BUILD_LIBTORCH
      at_npu::native::NpuUtils::ProfReportMarkDataToNpuProfiler(1, op_name, params.correlation_id);
#endif
    } else {
      this->func();
    }
  }

private:
  PROCESS_FUNC func = nullptr;
  string opName;
};

} // namespace native
} // namespace at_npu
