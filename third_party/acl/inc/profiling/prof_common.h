/**
 * @file prof_common.h
 *
 * Minimal common profiling declarations required by DVM and MLIR launcher code.
 */
#ifndef MSPROFILER_PROF_COMMON_H
#define MSPROFILER_PROF_COMMON_H

#include "aprof_pub.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MSPROF_DATA_HEAD_MAGIC_NUM 0x5A5AU
#define MSPROF_TASK_TIME_L0 0x00000800ULL

typedef const void *ConstVoidPtr;

#ifdef __cplusplus
}
#endif

#endif
