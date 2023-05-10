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

#ifndef HI_VDEC_H_
#define HI_VDEC_H_

#include "hi_dvpp_common.h"
#include "hi_dvpp_vb.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif // #ifdef __cplusplus

#define VDEC_MAX_CHN_NUM 256

/*********************************************************************************************/
/* Invalid channel ID. */
#define HI_ERR_VDEC_INVALID_CHN_ID 0xA0058002
/* At least one parameter is illegal ,eg, an illegal enumeration value. */
#define HI_ERR_VDEC_ILLEGAL_PARAM  0xA0058003
/* Channel exists. */
#define HI_ERR_VDEC_EXIST          0xA0058004
/* The channel does not exist. */
#define HI_ERR_VDEC_UNEXIST        0xA0058005
/* Using a NULL pointer. */
#define HI_ERR_VDEC_NULL_PTR       0xA0058006
/* Try to enable or initialize system, device or channel, before configure attribute. */
#define HI_ERR_VDEC_NOT_CFG        0xA0058007
/* Operation is not supported by NOW. */
#define HI_ERR_VDEC_NOT_SUPPORT    0xA0058008
/* Operation is not permitted, eg, try to change static attribute. */
#define HI_ERR_VDEC_NOT_PERM       0xA0058009
/* Failure caused by malloc memory. */
#define HI_ERR_VDEC_NO_MEM         0xA005800C
/* Failure caused by malloc buffer. */
#define HI_ERR_VDEC_NO_BUF         0xA005800D
/* No data in buffer. */
#define HI_ERR_VDEC_BUF_EMPTY      0xA005800E
/* No buffer for new data. */
#define HI_ERR_VDEC_BUF_FULL       0xA005800F
/* System is not ready, had not initialized or loaded. */
#define HI_ERR_VDEC_SYS_NOT_READY  0xA0058010
/* Bad address, eg. used for copy_from_user & copy_to_user. */
#define HI_ERR_VDEC_BAD_ADDR       0xA0058011
/* System busy */
#define HI_ERR_VDEC_BUSY           0xA0058012
/* system error */
#define HI_ERR_VDEC_SYS_ERROR      0xA0058015

typedef hi_s32 hi_vdec_chn;

typedef enum {
    HI_VDEC_SEND_MODE_STREAM = 0,  // Send by stream.
    HI_VDEC_SEND_MODE_FRAME,  // Send by frame.
    HI_VDEC_SEND_MODE_COMPAT, // One frame supports multiple packets sending.
    HI_VDEC_SEND_MODE_BUTT
} hi_vdec_send_mode;

typedef enum {
    HI_VIDEO_DEC_MODE_IPB = 0,
    HI_VIDEO_DEC_MODE_IP,
    HI_VIDEO_DEC_MODE_I,
    HI_VIDEO_DEC_MODE_BUTT
} hi_video_dec_mode;

typedef enum {
    HI_VDEC_FRAME_TYPE_I = 0,
    HI_VDEC_FRAME_TYPE_P = 1,
    HI_VDEC_FRAME_TYPE_B = 2,
    HI_VDEC_FRAME_TYPE_BUTT
} hi_vdec_frame_type;

typedef enum {
    HI_QUICK_MARK_ADAPT = 0,
    HI_QUICK_MARK_FORCE,
    HI_QUICK_MARK_NONE,
    HI_QUICK_MARK_BUTT
} hi_quick_mark_mode;

typedef struct {
    hi_vdec_frame_type frame_type;
    hi_u32 err_rate;
    hi_u32 poc;
} hi_vdec_video_supplement_info;

typedef struct {
    hi_payload_type type;  // RW; Video type to be decoded.
    union {
        hi_vdec_video_supplement_info video_supplement_info; // Structure with video (h265/h264)
    };
} hi_vdec_supplement_info;

typedef struct {
    hi_s32 set_pic_size_err;  // R; Picture width or height is larger than channel width or height.
    hi_s32 set_protocol_num_err;  // R; Protocol num is not enough. eg: slice, pps, sps.
    hi_s32 set_ref_num_err;  // R; Reference num is not enough.
    hi_s32 set_pic_buf_size_err;  // R; The buffer size of picture is not enough.
    hi_s32 format_err;  // R; Format error. eg: do not support filed.
    hi_s32 stream_unsupport;  // R; Unsupported the stream specification.
    hi_s32 pack_err;  // R; Stream package error.
    hi_s32 stream_size_over;  // R; The stream size is too big and force discard stream.
    hi_s32 stream_not_release;  // R; The stream not released for too long time.
} hi_vdec_dec_err;

typedef struct {
    hi_payload_type type;  // R; Video type to be decoded.
    hi_u32 left_stream_bytes;  // R; Left stream bytes waiting for decode.
    hi_u32 left_stream_frames;  // R; Left frames waiting for decode,only valid for HI_VDEC_SEND_MODE_FRAME.
    hi_u32 left_decoded_frames;  // R; Pics waiting for output.
    hi_bool is_started;  // R; Had started recv stream?
    hi_u32 recv_stream_frames;  // R; How many frames of stream has been received. valid when send by frame.
    hi_u32 dec_stream_frames;  // R; How many frames of stream has been decoded. valid when send by frame.
    hi_vdec_dec_err dec_err;  // R; Information about decode error.
    hi_u32 width;  // R; The width of the currently decoded stream.
    hi_u32 height;  // R; The height of the currently decoded stream.
    hi_u64 latest_frame_pts;  // R; PTS of the latest decoded frame.
} hi_vdec_chn_status;

typedef struct {
    hi_u32 ref_frame_num;  // RW, Range: [0, 16]; reference frame num.
    hi_bool temporal_mvp_en; // RW; Specifies whether temporal motion vector predictors.
    hi_u32 tmv_buf_size;  // RW; The size of tmv buffer (byte).
} hi_vdec_video_attr;

typedef struct {
    hi_payload_type type;  // RW; Video type to be decoded.
    hi_vdec_send_mode mode;  // RW; Send by stream or by frame.
    hi_u32 pic_width;  // RW; Max width of pic.
    hi_u32 pic_height;  // RW; Max height of pic.
    hi_u32 stream_buf_size;  // RW; The size of stream buffer (byte).
    hi_u32 frame_buf_size;  // RW; The size of frame buffer (byte).
    hi_u32 frame_buf_cnt;
    union {
        hi_vdec_video_attr video_attr; // RW; Structure with video (h264/h265).
    };
} hi_vdec_chn_attr;

typedef struct {
    hi_bool end_of_frame;  // W; Is the end of a frame.
    hi_bool end_of_stream;  // W; Is the end of all stream.
    hi_bool need_display;  // W; Is the current frame displayed. only valid by HI_VDEC_SEND_MODE_FRAME.
    hi_u64 pts;  // W; Time stamp.
    hi_u64 private_data;  // W; Private data, only valid by HI_VDEC_SEND_MODE_FRAME or HI_VDEC_SEND_MODE_COMPAT.
    hi_u32 len;  // W; The len of stream.
    hi_u8 *ATTRIBUTE addr;  // W; The address of stream.
} hi_vdec_stream;

typedef struct {
    // RW; HI_FALSE: output base layer; HI_TRUE: output enhance layer; default: HI_FALSE
    hi_bool composite_dec_en;
    // RW; HI_FALSE: don't support slice low latency; HI_TRUE: support slice low latency; default: HI_FALSE
    hi_bool slice_input_en;
    // RW, Range: [0, 100]; threshold for stream error process,
    // 0: discard with any error, 100 : keep data with any error.
    hi_s32 err_threshold;
    // RW; Decode mode , 0: decode IPB frames,
    // 1: only decode I frame & P frame , 2: only decode I frame.
    hi_video_dec_mode dec_mode;
    // RW; Frames output order ,
    // 0: the same with display order , 1: the same width decoder order.
    hi_video_out_order out_order;
    hi_compress_mode compress_mode;  // RW; Compress mode.
    hi_video_format video_format;  // RW; Video format.
    hi_quick_mark_mode quick_mark_mode;
} hi_vdec_video_param;

typedef struct {
    hi_pixel_format pixel_format;  // RW; Out put pixel format.
    // RW, range: [0, 255]; Value 0 is transparent.
    // [0 ,127]   is deemed to transparent when pixel_format is ARGB1555 or ABGR1555
    // [128 ,256] is deemed to non-transparent when pixel_format is ARGB1555 or ABGR1555.
    hi_u32 alpha;
} hi_vdec_pic_param;

typedef struct {
    hi_payload_type type; // RW; video type to be decoded
    hi_u32 display_frame_num; // RW, Range: [0, 16]; display frame num
    union {
        hi_vdec_video_param video_param; // structure with video ( h265/h264)
        hi_vdec_pic_param pic_param; // structure with picture (jpeg/mjpeg)
    };
} hi_vdec_chn_param;

typedef struct {
    hi_u32 width;
    hi_u32 height;
    hi_u32 width_stride;
    hi_u32 height_stride;
    hi_pixel_format pixel_format; // JPEGD/VDEC OutPut Format
    hi_u64 vir_addr;
    hi_u32 buffer_size;
    hi_s16 offset_top; // For JPEGD Region Decoding
    hi_s16 offset_bottom; // For JPEGD Region Decoding
    hi_s16 offset_left; // For JPEGD Region Decoding
    hi_s16 offset_right; // For JPEGD Region Decoding
} hi_vdec_pic_info;

typedef struct {
    hi_s32 max_slice_num; // RW; max slice num support
    hi_s32 max_sps_num; // RW; max sps num support
    hi_s32 max_pps_num; // RW; max pps num support
} hi_h264_protocol_param;

typedef struct {
    hi_s32 max_slice_segment_num; // RW; max slice segmnet num support
    hi_s32 max_vps_num; // RW; max vps num support
    hi_s32 max_sps_num; // RW; max sps num support
    hi_s32 max_pps_num; // RW; max pps num support
} hi_h265_protocol_param;

typedef struct {
    hi_payload_type type; // RW; video type to be decoded, only h264 and h265 supported
    union {
        hi_h264_protocol_param h264_param; // protocol param structure for h264
        hi_h265_protocol_param h265_param; // protocol param structure for h265
    };
} hi_vdec_protocol_param;

typedef enum {
    YUVOUT_ALIGN_DOWN = 0,
    YUVOUT_ALIGN_UP = 1,
    YUVOUT_ALIGN_DOWN_COMPAT = 2,
} hi_jpegd_precision_mode;

/*
 * @brief get video decoder picture buffer size
 * @param [in] type: video type [PT_H264/PT_H265]
 * @param [in] buf_attr: pointer of picture buffer attribute
 * @return : video decoder picture buffer size
 */
hi_u32 hi_vdec_get_pic_buf_size(hi_payload_type type, hi_pic_buf_attr *buf_attr);

/*
 * @brief get video decoder tmv buffer size
 * @param [in] type: video type [PT_H264/PT_H265]
 * @param [in] width: video width
 * @param [in] height: video height
 * @return : video decoder tmv buffer size
 */
hi_u32 hi_vdec_get_tmv_buf_size(hi_payload_type type, hi_u32 width, hi_u32 height);

/*
 * @brief create video decoder channel
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [in] attr: pointer of video decoder channel attribute
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_create_chn(hi_vdec_chn chn, const hi_vdec_chn_attr *attr);

/*
 * @brief destroy video decoder channel
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_destroy_chn(hi_vdec_chn chn);

/*
 * @brief set video decoder channel attribute
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [in] attr: pointer of video decoder channel attribute
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_set_chn_attr(hi_vdec_chn chn, const hi_vdec_chn_attr *attr);

/*
 * @brief get video decoder channel attribute
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [out] attr: pointer of video decoder channel attribute
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_get_chn_attr(hi_vdec_chn chn, hi_vdec_chn_attr *attr);

/*
 * @brief set video decoder channel protocol param
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [in] protocol_param: pointer of video decoder protocol param
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_set_protocol_param(hi_vdec_chn chn, const hi_vdec_protocol_param *protocol_param);

/*
 * @brief get video decoder channel protocol param
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [out] protocol_param: pointer of video decoder protocol param
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_get_protocol_param(hi_vdec_chn chn, hi_vdec_protocol_param *protocol_param);

/*
 * @brief video decoder channel start receive stream
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_start_recv_stream(hi_vdec_chn chn);

/*
 * @brief video decoder channel stop receive stream
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_stop_recv_stream(hi_vdec_chn chn);

/*
 * @brief query video decoder channel status
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [out] status: pointer of video decoder channel status struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_query_status(hi_vdec_chn chn, hi_vdec_chn_status *status);

/*
 * @brief reset video decoder channel status
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_reset_chn(hi_vdec_chn chn);

/*
 * @brief set video decoder channel parameter
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [in] chn_param: pointer of video decoder channel parameter struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_set_chn_param(hi_vdec_chn chn, const hi_vdec_chn_param *chn_param);

/*
 * @brief get video decoder channel parameter
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [out] chn_param: pointer of video decoder channel parameter struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_get_chn_param(hi_vdec_chn chn, hi_vdec_chn_param *chn_param);

/*
 * @brief send stream and outbuffer to video decoder channel
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [in] stream: pointer of stream struct
 * @param [in] vdec_pic_info: pointer of vdec_pic_info struct
 * @param [in] milli_sec: -1 is block,0 is no block,other positive number is timeout
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_send_stream(hi_vdec_chn chn, const hi_vdec_stream *stream, hi_vdec_pic_info *vdec_pic_info,
    hi_s32 milli_sec);

/*
 * @brief get frame from video decoder channel
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [in] milli_sec: -1 is block,0 is no block,other positive number is timeout
 * @param [out] frame_info: pointer of frame info struct
 * @param [out] supplement: pointer of supplement info struct
 * @param [out] stream: pointer of stream struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_get_frame(hi_vdec_chn chn, hi_video_frame_info *frame_info,  hi_vdec_supplement_info *supplement,
                             hi_vdec_stream *stream, hi_s32 milli_sec);

/*
 * @brief release frame from video decoder channel
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [in] frame_info: pointer of frame info struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_release_frame(hi_vdec_chn chn, const hi_video_frame_info *frame_info);

/*
 * @brief get video decoder channel fd
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @return success: return fd
 *         fail: return negative number
 */
hi_s32 hi_mpi_vdec_get_fd(hi_vdec_chn chn);

/*
 * @brief close video decoder channel fd
 * @param [in] chn: video decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @return success: return 0
 *         fail: return -1
 */
hi_s32 hi_mpi_vdec_close_fd(hi_vdec_chn chn);

/*
 * @brief get input image's information parsed by dvpp
 * @param [in] img_type: payload type of input image
 * @param [in] stream: stream info pointer
 * @param [out] img_info: parsed image info pointer
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_dvpp_get_image_info(hi_payload_type img_type, const hi_vdec_stream *stream, hi_img_info *img_info);

/*
 * @brief set jpegd decoder channel precision mode
 * @param [in] chn: jpegd decoder channel id [0, VDEC_MAX_CHN_NUM)
 * @param [in] mode: jpegd decoder channel precision mode struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vdec_set_jpegd_precision_mode(hi_vdec_chn chn, const hi_jpegd_precision_mode mode);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif // #ifdef __cplusplus

#endif // #ifndef HI_VDEC_H_
