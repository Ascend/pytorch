/**
* @file aml_fwk_detect.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2023-2024. All Rights Reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef INC_FWK_AML_FWK_DETECT_H_
#define INC_FWK_AML_FWK_DETECT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t AmlStatus;

typedef enum AmlDetectRunMode {
    AML_DETECT_RUN_MODE_ONLINE = 0,
    AML_DETECT_RUN_MODE_OFFLINE = 1,
    AML_DETECT_RUN_MODE_MAX,
} AmlDetectRunMode;

typedef struct AmlAicoreDetectAttr {
    AmlDetectRunMode mode;
    void *workspace;
    uint64_t workspaceSize;
    uint8_t reserve[64];
} AmlAicoreDetectAttr;

AmlStatus AmlAicoreDetectOnline(int32_t deviceId, const AmlAicoreDetectAttr *attr);

#ifdef __cplusplus
}
#endif

#endif // INC_FWK_AML_FWK_DETECT_H_
