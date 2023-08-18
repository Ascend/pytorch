#ifndef __TORCH_NPU_MSPROFILERINTERFACE__
#define __TORCH_NPU_MSPROFILERINTERFACE__

#include <third_party/acl/inc/acl/acl_msprof.h>
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace at_npu {
namespace native {


void *AclprofCreateStamp();

void AclprofDestroyStamp(void *stamp);

aclError AclprofSetStampTagName(void *stamp, const char *tagName, uint16_t len);

aclError AclprofSetCategoryName(uint32_t category, const char *categoryName);

aclError AclprofSetStampCategory(void *stamp, uint32_t category);

aclError AclprofSetStampPayload(void *stamp, const int32_t type, void *value);

aclError AclprofSetStampTraceMessage(void *stamp, const char *msg, uint32_t msgLen);

aclError AclprofSetStampCallStack(void *stamp, const char *callStack, uint32_t len);

aclError AclprofMsproftxSwitch(bool isOpen);

aclError AclprofMark(void *stamp);

aclError AclprofPush(void *stamp);

aclError AclprofPop();

aclError AclprofRangeStart(void *stamp, uint32_t *rangeId);

aclError AclprofRangeStop(uint32_t rangeId);

aclError AclprofReportStamp(const char *tag, unsigned int tagLen,
                            unsigned char *data, unsigned int dataLen);

bool CheckInterfaceReportStamp();

aclError AclprofSetConfig(aclprofConfigType configType, const char* config, size_t configLength);

}
}

#endif // __TORCH_NPU_MSPROFILERINTERFACE__
