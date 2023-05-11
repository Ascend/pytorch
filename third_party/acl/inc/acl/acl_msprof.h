/**
* @file acl_msprof.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef INC_EXTERNAL_ACL_ACL_MSPROF_H_
#define INC_EXTERNAL_ACL_ACL_MSPROF_H_

#include "acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif


ACL_FUNC_VISIBILITY void *aclprofCreateStamp();

/**

@ingroup AscendCL
@yutong zhang destroy aclprofStamp pointer
@retval void
*/
ACL_FUNC_VISIBILITY void aclprofDestroyStamp(void *stamp);

ACL_FUNC_VISIBILITY aclError aclprofSetStampTagName(void *stamp, const char *tagName, uint16_t len);

ACL_FUNC_VISIBILITY aclError aclprofSetCategoryName(uint32_t category, const char *categoryName);

ACL_FUNC_VISIBILITY aclError aclprofSetStampCategory(void *stamp, uint32_t category);

ACL_FUNC_VISIBILITY aclError aclprofSetStampPayload(void *stamp, const int32_t type, void *value);

ACL_FUNC_VISIBILITY aclError aclprofSetStampTraceMessage(void *stamp, const char *msg, uint32_t msgLen);

ACL_FUNC_VISIBILITY aclError aclprofSetStampCallStack(void *stamp, const char *callStack, uint32_t len);

ACL_FUNC_VISIBILITY aclError aclprofMsproftxSwitch(bool isOpen);

ACL_FUNC_VISIBILITY aclError aclprofMark(void *stamp);

ACL_FUNC_VISIBILITY aclError aclprofPush(void *stamp);

ACL_FUNC_VISIBILITY aclError aclprofPop();

ACL_FUNC_VISIBILITY aclError aclprofRangeStart(void *stamp, uint32_t *rangeId);

ACL_FUNC_VISIBILITY aclError aclprofRangeStop(uint32_t rangeId);

ACL_FUNC_VISIBILITY aclError aclprofReportStamp(
    const char *tag, unsigned int tagLen, unsigned char *data, unsigned int dataLen);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_MSPROF_H_
