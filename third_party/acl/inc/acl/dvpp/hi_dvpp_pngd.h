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

#ifndef HI_PNGD_H_
#define HI_PNGD_H_

#include "hi_dvpp_common.h"
#include "hi_dvpp_vb.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif // #ifdef __cplusplus

#define PNGD_MAX_CHN_NUM 128

/*********************************************************************************************/
/* Invalid channel ID. */
#define HI_ERR_PNGD_INVALID_CHN_ID 0xA0408002
/* At least one parameter is illegal ,eg, an illegal enumeration value. */
#define HI_ERR_PNGD_ILLEGAL_PARAM  0xA0408003
/* Channel exists. */
#define HI_ERR_PNGD_EXIST          0xA0408004
/* The channel does not exist. */
#define HI_ERR_PNGD_UNEXIST        0xA0408005
/* Using a NULL pointer. */
#define HI_ERR_PNGD_NULL_PTR       0xA0408006
/* Try to enable or initialize system, device or channel, before configure attribute. */
#define HI_ERR_PNGD_NOT_CFG        0xA0408007
/* Operation is not supported by NOW. */
#define HI_ERR_PNGD_NOT_SUPPORT    0xA0408008
/* Operation is not permitted, eg, try to change static attribute. */
#define HI_ERR_PNGD_NOT_PERM       0xA0408009
/* Failure caused by malloc memory. */
#define HI_ERR_PNGD_NO_MEM         0xA040800C
/* Failure caused by malloc buffer. */
#define HI_ERR_PNGD_NO_BUF         0xA040800D
/* No data in buffer. */
#define HI_ERR_PNGD_BUF_EMPTY      0xA040800E
/* No buffer for new data. */
#define HI_ERR_PNGD_BUF_FULL       0xA040800F
/* System is not ready, had not initialized or loaded. */
#define HI_ERR_PNGD_SYS_NOT_READY  0xA0408010
/* Bad address, eg. used for copy_from_user & copy_to_user. */
#define HI_ERR_PNGD_BAD_ADDR       0xA0408011
/* System busy */
#define HI_ERR_PNGD_BUSY           0xA0408012
/* hardware or software timeout */
#define HI_ERR_PNGD_TIMEOUT        0xA0408014
/* Internal system error. */
#define HI_ERR_PNGD_SYS_ERROR      0xA0408015

typedef hi_s32 hi_pngd_chn;

typedef struct {
    hi_u32 stream_que_cnt;  // reserved
    hi_u64 reserved[4];
} hi_pngd_chn_attr;

/*
 * @brief create png decoder channel
 * @param [in] chn: png decoder channel id [0, PNGD_MAX_CHN_NUM)
 * @param [in] attr: pointer of png decoder channel attribute
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_pngd_create_chn(hi_pngd_chn chn, const hi_pngd_chn_attr *attr);

/*
 * @brief destroy png decoder channel
 * @param [in] chn: png decoder channel id [0, PNGD_MAX_CHN_NUM)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_pngd_destroy_chn(hi_pngd_chn chn);

/*
 * @brief send stream and outbuffer to png decoder channel
 * @param [in] chn: png decoder channel id [0, PNGD_MAX_CHN_NUM)
 * @param [in] stream: pointer of stream struct
 * @param [in] png_pic_info: pointer of hi_pic_info struct
 * @param [in] milli_sec: -1 is block,0 is no block,other positive number is timeout
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_pngd_send_stream(hi_pngd_chn chn, const hi_img_stream *stream, hi_pic_info *png_pic_info,
    hi_s32 milli_sec);

/*
 * @brief get frame from png decoder channel
 * @param [in] chn: png decoder channel id [0, PNGD_MAX_CHN_NUM)
 * @param [in] milli_sec: -1 is block,0 is no block,other positive number is timeout
 * @param [out] png_pic_info: pointer of pic info struct
 * @param [out] stream: pointer of stream struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_pngd_get_image_data(hi_pngd_chn chn, hi_pic_info *png_pic_info, hi_img_stream *stream, hi_s32 milli_sec);

/*
 * @brief get input image's information parsed by dvpp
 * @param [in] stream: stream info pointer
 * @param [out] img_info: parsed image info pointer
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_png_get_image_info(const hi_img_stream *png_stream, hi_img_info *img_info);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif // #ifdef __cplusplus

#endif // #ifndef HI_PNGD_H_
