#ifndef __TORCH_NPU_MSPROFILERINTERFACE__
#define __TORCH_NPU_MSPROFILERINTERFACE__

#include <third_party/acl/inc/acl/acl_msprof.h>
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace native {


aclError AclprofSetConfig(aclprofConfigType configType, const char* config, size_t configLength);

aclError AclprofGetSupportedFeatures(size_t* featuresSize, void** featuresData);

aclError AclProfilingMarkEx(const char *msg, size_t msgLen, aclrtStream stream);
}
}

#endif // __TORCH_NPU_MSPROFILERINTERFACE__
