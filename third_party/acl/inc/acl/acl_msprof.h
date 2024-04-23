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

typedef enum {
    ACL_PROF_ARGS_MIN = 0,
    ACL_PROF_STORAGE_LIMIT,
    ACL_PROF_AIV_METRICS,
    ACL_PROF_SYS_HARDWARE_MEM_FREQ,
    ACL_PROF_LLC_MODE,
    ACL_PROF_SYS_IO_FREQ,
    ACL_PROF_SYS_INTERCONNECTION_FREQ,
    ACL_PROF_DVPP_FREQ,
    ACL_PROF_HOST_SYS,
    ACL_PROF_HOST_SYS_USAGE,
    ACL_PROF_HOST_SYS_USAGE_FREQ,
    ACL_PROF_ARGS_MAX,
} aclprofConfigType;

ACL_FUNC_VISIBILITY aclError aclprofSetConfig(aclprofConfigType configType, const char* config, size_t configLength);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_ACL_ACL_MSPROF_H_
