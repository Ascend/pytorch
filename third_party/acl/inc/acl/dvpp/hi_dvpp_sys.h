/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 and
 * only version 2 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * Description:
 * Author: huawei
 * Create: 2020-4-1
 */

#ifndef HI_DVPP_SYS_H_
#define HI_DVPP_SYS_H_

#include "hi_dvpp_common.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#define HI_ERR_SYS_ILLEGAL_PARAM 0xA0028003
#define HI_ERR_SYS_EXIST         0xA0028004
#define HI_ERR_SYS_UNEXIST       0xA0028005
#define HI_ERR_SYS_NULL_PTR      0xA0028006
#define HI_ERR_SYS_NOT_SUPPORT   0xA0028008
#define HI_ERR_SYS_NOT_PERM      0xA0028009
#define HI_ERR_SYS_NO_MEM        0xA002800C
#define HI_ERR_SYS_NOT_READY     0xA0028010
#define HI_ERR_SYS_BUSY          0xA0028012
#define HI_ERR_SYS_ERR           0xA0028015

typedef enum {
    HI_DVPP_EPOLL_CTL_ADD = 1,
    HI_DVPP_EPOLL_CTL_MOD = 2,
    HI_DVPP_EPOLL_CTL_DEL = 3,
    HI_DVPP_EPOLL_CTL_BUTT
} hi_dvpp_epoll_ctl_op;

typedef enum {
    HI_DVPP_EPOLL_IN = 1u,
    HI_DVPP_EPOLL_OUT = 1u << 1u,
    HI_DVPP_EPOLL_ET = (hi_u32)1u << 31u
} hi_dvpp_epoll_event_type;

typedef struct {
    hi_u32 events;
    hi_void *data;
} hi_dvpp_epoll_event;

/*
 * @brief init mpp system
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_sys_init(hi_void);

/*
 * @brief exit mpp system
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_sys_exit(hi_void);

/**
 * @brief Dvpp Epoll create interface
 * @param [in] size: Specify the number of dvpp channel descriptors to monitor,
 *                      currently ignored, and must be positive
 * @param [out] epoll_fd: return the descriptor referring to the new dvpp epoll instance
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_sys_create_epoll(hi_s32 size, hi_s32 *epoll_fd);

/**
 * @brief Dvpp Epoll control interface
 * @param [in] epoll_fd: descriptor of dvpp epoll instance
 * @param [in] operation: operation type for the target dvpp channel descriptor referred by @s32Fd
 * @param [in] fd: descriptor of target dvpp channel
 * @param [in] event: describes the event type wants to monitor, and also the data go with the channel descriptor
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_sys_ctl_epoll(hi_s32 epoll_fd, hi_s32 operation, hi_s32 fd, hi_dvpp_epoll_event *event);

/**
 * @brief Dvpp Epoll wait interface
 * @param [in] epoll_fd: descriptor of dvpp epoll instance
 * @param [out] events: return events that happened
 * @param [in] max_events: maxinum numer of events can return
 * @param [in] timeout: milliseconds the caller can block
 * @param [out] event_num: return the numer of events saved in @pEvents
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_sys_wait_epoll(hi_s32 epoll_fd,
    hi_dvpp_epoll_event *events, hi_s32 max_events, hi_s32 timeout, hi_s32 *event_num);

/**
 * @brief Dvpp Epoll close interface
 * @param [in] epoll_fd:  descriptor of dvpp epoll instance
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_sys_close_epoll(hi_s32 epoll_fd);

/*
 * @brief set csc matrix coefficient
 * @param [in] mode: dvpp module
 * @param [in] chn: chn id
 * @param [in] csc_matrix: csc matrix mode
 * @param [in] csc_coefficient: csc matrix coefficient when use user mode
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_sys_set_chn_csc_matrix(hi_mod_id mode,
    hi_s32 chn, hi_csc_matrix csc_matrix, hi_csc_coefficient *csc_coefficient);

/*
 * @brief get csc matrix coefficient
 * @param [in] mode: dvpp module
 * @param [in] chn: chn id
 * @param [out] csc_matrix: csc matrix mode
 * @param [out] csc_coefficient: csc matrix coefficient for all mode
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_sys_get_chn_csc_matrix(hi_mod_id mode,
    hi_s32 chn, hi_csc_matrix *csc_matrix, hi_csc_coefficient *csc_coefficient);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif // #ifndef HI_DVPP_SYS_H_
