#ifndef __PULGIN_NATIVE_NPU_UTILS_NUP_UTILS__
#define __PULGIN_NATIVE_NPU_UTILS_NUP_UTILS__

#include <stdint.h>
#include <string>
#include <vector>
#include <ATen/ATen.h>
#include "torch_npu/csrc/core/npu/npu_log.h"

#include "third_party/acl/inc/ge/ge_error_codes.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_op.h"

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"

using std::string;
using std::vector;

namespace at_npu
{
  namespace native
  {

    // smallvector max size
    const int N = 32;
    // npu tensor max size
    const int SHAPE_SIZE = 8;
    // HALF_MAX and HALF_MIN of NPU support
    const int NPU_HALF_MAX = 65504;
    const int NPU_HALF_MIN = -65504;
    const int NPU_MAX_OP_EXEC_TRY_NUM = 2;

    typedef enum CompileType
    {
      MEMORY_HOST_COMPILE_DEPENDENT = 1,
      MEMORY_HOST_COMPILE_INDEPENDENT = 2,
    } CompileType;

    class NpuUtils
    {
    public:
      static bool check_match(const at::Tensor *tensor);
      TORCH_NPU_API static at::Tensor format_contiguous(const at::Tensor &src);
      static at::Tensor format_contiguous_add_copy_optimize(const at::Tensor &src);
      static void RefreshFormat(const at::Tensor &tensor);
      static void format_fresh_view(
          at::Tensor &x,
          const at::Tensor &y);

      static bool check_5d_5d_match(const at::Tensor &tensor);
      static bool IsOomError(aclError ret, int index);
      static void check_1d(const at::Tensor &t, const char *arg, const char *fn);
#ifndef BUILD_LIBTORCH
      static void ProfReportMarkData(const std::string &msg);
      static void ProfReportMarkDataToNpuProfiler(uint32_t category, const std::string &data, uint64_t correlation_id = 0);
      static void ProfReportMarkDataToNpuProfiler(uint32_t category, void *data, size_t offset);
#endif
    private:
      using DqueueCall = void (*)(c10_npu::queue::QueueParas *para, uint32_t category);
      static void DqueueCompileExcute(c10_npu::queue::QueueParas *para, uint32_t category);
      static void DqueueAnyncMemcpy(c10_npu::queue::QueueParas *para, uint32_t category);
      static void DqueueEvent(c10_npu::queue::QueueParas *para, uint32_t category);
      static void DqueueCompileExcuteBs(c10_npu::queue::QueueParas *para, uint32_t category);
    };
    const std::string AclDateTypeToString(aclDataType descDType);
    const std::string AclFormatToString(aclFormat descFormat);
  } // namespace native
} // namespace at_npu

#endif // __NATIVE_NPU_UTILS_NUP_UTILS__
