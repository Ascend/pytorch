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

#ifndef HI_VENC_H_
#define HI_VENC_H_

#include "hi_dvpp_common.h"
#include "hi_dvpp_vb.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#define HI_VENC_MAX_PACK_INFO_NUM 8
#define HI_VENC_MAX_MPF_NUM 2
#define HI_VENC_PRORES_MAX_ID_CHAR_NUM 4
#define HI_VENC_JPEG_QT_COEF_NUM 64
#define HI_VENC_QP_HIST_NUM 52
#define HI_VENC_MAX_SSE_NUM 8
#define HI_VENC_TEXTURE_THRESHOLD_SIZE 16 // The number of thresholds that define the complexity of RC macroblocks

/* invlalid channel ID */
#define HI_ERR_VENC_INVALID_CHN_ID 0xA0088002
/* at least one parameter is illagal ,eg, an illegal enumeration value  */
#define HI_ERR_VENC_ILLEGAL_PARAM  0xA0088003
/* channel exists */
#define HI_ERR_VENC_EXIST          0xA0088004
/* channel exists */
#define HI_ERR_VENC_UNEXIST        0xA0088005
/* using a NULL pointer */
#define HI_ERR_VENC_NULL_PTR       0xA0088006
/* try to enable or initialize system,device or channel, before configing attribute */
#define HI_ERR_VENC_NOT_CFG        0xA0088007
/* operation is not supported by NOW */
#define HI_ERR_VENC_NOT_SUPPORT    0xA0088008
/* operation is not permitted ,eg, try to change statuses attribute */
#define HI_ERR_VENC_NOT_PERM       0xA0088009
/* failure caused by malloc memory */
#define HI_ERR_VENC_NO_MEM         0xA008800C
/* failure caused by malloc buffer */
#define HI_ERR_VENC_NO_BUF         0xA008800D
/* no data in buffer */
#define HI_ERR_VENC_BUF_EMPTY      0xA008800E
/* no buffer for new data */
#define HI_ERR_VENC_BUF_FULL       0xA008800F
/* system is not ready,had not initialed or loaded */
#define HI_ERR_VENC_SYS_NOT_READY  0xA0088010
/* bad address, eg, used for copy_from_user & copy_to_user */
#define HI_ERR_VENC_BAD_ADDR       0xA0088011
/* system is busy */
#define HI_ERR_VENC_BUSY           0xA0088012
/* system err */
#define HI_ERR_VENC_SYS_ERROR      0xA0088015

typedef hi_s32 hi_venc_chn;

// the sse info
typedef struct {
    hi_bool enable; // RW; Range:[0,1]; Region SSE enable
    hi_u64  sse_val; // R; Region SSE value
} hi_venc_sse_info;

// the nalu type of H264E
typedef enum {
    HI_VENC_H264_NALU_B_SLICE = 0, // B SLICE types
    HI_VENC_H264_NALU_P_SLICE = 1, // P SLICE types
    HI_VENC_H264_NALU_I_SLICE = 2, // I SLICE types
    HI_VENC_H264_NALU_IDR_SLICE = 5, // IDR SLICE types
    HI_VENC_H264_NALU_SEI    = 6, // SEI types
    HI_VENC_H264_NALU_SPS    = 7, // SPS types
    HI_VENC_H264_NALU_PPS    = 8, // PPS types
    HI_VENC_H264_NALU_BUTT
} hi_venc_h264_nalu_type;

// the nalu type of H265E
typedef enum {
    HI_VENC_H265_NALU_B_SLICE = 0, // B SLICE types
    HI_VENC_H265_NALU_P_SLICE = 1, // P SLICE types
    HI_VENC_H265_NALU_I_SLICE = 2, // I SLICE types
    HI_VENC_H265_NALU_IDR_SLICE = 19, // IDR SLICE types
    HI_VENC_H265_NALU_VPS    = 32, // VPS types
    HI_VENC_H265_NALU_SPS    = 33, // SPS types
    HI_VENC_H265_NALU_PPS    = 34, // PPS types
    HI_VENC_H265_NALU_SEI    = 39, // SEI types
    HI_VENC_H265_NALU_ENHANCE   = 64, // ENHANCE types
    HI_VENC_H265_NALU_BUTT
} hi_venc_h265_nalu_type;

// the reference type of H264E slice
typedef enum {
    HI_VENC_H264_REF_SLICE_FOR_1X = 1, // Reference slice for H264E_REF_MODE_1X
    HI_VENC_H264_REF_SLICE_FOR_2X = 2, // Reference slice for H264E_REF_MODE_2X
    HI_VENC_H264_REF_SLICE_FOR_4X = 5, // Reference slice for H264E_REF_MODE_4X
    HI_VENC_H264_REF_SLICE_FOR_BUTT // slice not for reference
} hi_venc_h264_ref_slice_type;

// the pack type of JPEGE
typedef enum {
    HI_VENC_JPEG_PACK_ECS = 5, // ECS types
    HI_VENC_JPEG_PACK_APP = 6, // APP types
    HI_VENC_JPEG_PACK_VDO = 7, // VDO types
    HI_VENC_JPEG_PACK_PIC = 8, // PIC types
    HI_VENC_JPEG_PACK_DCF = 9, // DCF types
    HI_VENC_JPEG_PACK_DCF_PIC = 10, // DCF PIC types
    HI_VENC_JPEG_PACK_BUTT
} hi_venc_jpege_pack_type;

// the pack type of PRORES
typedef enum {
    HI_VENC_PRORES_PACK_PIC = 1, // PIC types
    HI_VENC_PRORES_PACK_BUTT
} hi_venc_prores_pack_type;

// the data type of VENC
typedef union {
    hi_venc_h264_nalu_type    h264_type; // R; H264E NALU types
    hi_venc_jpege_pack_type    jpeg_type; // R; JPEGE pack types
    hi_venc_h265_nalu_type    h265_type; // R; H264E NALU types
    hi_venc_prores_pack_type   prores_type;
} hi_venc_data_type;

// the pack info of VENC
typedef struct {
    hi_venc_data_type  pack_type; // R; the pack type
    hi_u32 pack_offset;
    hi_u32 pack_len;
} hi_venc_pack_info;

// Defines a stream packet
typedef struct {
    union {
        hi_u64           phys_addr; // R; the physics address of stream
        hi_u64           input_addr; // R; the address of input frame
    };
    hi_u8                ATTRIBUTE *addr; // R; the virtual address of stream
    hi_u32               ATTRIBUTE len; // R; the length of stream

    hi_u64               pts; // R; PTS
    hi_bool              is_frame_end; // R; frame end

    hi_venc_data_type    data_type; // R; the type of stream
    hi_u32               offset; // R; the offset between the Valid data and the start address
    hi_u32               data_num; // R; the  stream packets num
    hi_venc_pack_info    pack_info[HI_VENC_MAX_PACK_INFO_NUM]; // R; the stream packet Information
} hi_venc_pack;

// Defines the frame type and reference attributes of the H.264 frame skipping reference streams
typedef enum {
    HI_VENC_BASE_IDR_SLICE = 0, // the Idr frame at Base layer
    HI_VENC_BASE_P_SLICE_REF_TO_IDR, // the P frame at Base layer, referenced by other frames at Base layer and reference to Idr frame
    HI_VENC_BASE_P_SLICE_REF_BY_BASE, // the P frame at Base layer, referenced by other frames at Base layer
    HI_VENC_BASE_P_SLICE_REF_BY_ENHANCE, // the P frame at Base layer, referenced by other frames at Enhance layer
    HI_VENC_ENHANCE_P_SLICE_REF_BY_ENHANCE, // the P frame at Enhance layer, referenced by other frames at Enhance layer
    HI_VENC_ENHANCE_P_SLICE_NOT_FOR_REF, // the P frame at Enhance layer ,not referenced
    HI_VENC_P_SLICE_BUTT
} hi_venc_ref_type;

// Defines the features of an H.264 stream
typedef struct {
    hi_u32                 pic_bytes; // R; the coded picture stream byte number
    hi_u32                 inter16x16_mb_num; // R; the inter16x16 macroblock num
    hi_u32                 inter8x8_mb_num; // R; the inter8x8 macroblock num
    hi_u32                 intra16_mb_num; // R; the intra16x16 macroblock num
    hi_u32                 intra8_mb_num; // R; the intra8x8 macroblock num
    hi_u32                 intra4_mb_num; // R; the inter4x4 macroblock num

    hi_venc_ref_type       ref_type; // R; Type of encoded frames in advanced frame skipping reference mode
    hi_u32                 update_attr_cnt; // R; Number of times that channel attributes or
                                             // parameters(including RC parameters) are set
    hi_u32                 start_qp; // R; the start Qp of encoded frames
    hi_u32                 mean_qp; // R; the mean Qp of encoded frames
    hi_bool                is_p_skip;
} hi_venc_h264_stream_info;

// Defines the features of an H.265 stream
typedef struct {
    hi_u32                 pic_bytes; // R; the coded picture stream byte number
    hi_u32                 inter64x64_cu_num; // R; the inter64x64 cu num
    hi_u32                 inter32x32_cu_num; // R; the inter32x32 cu num
    hi_u32                 inter16x16_cu_num; // R; the inter16x16 cu num
    hi_u32                 inter8x8_cu_num; // R; the inter8x8   cu num
    hi_u32                 intra32x32_cu_num; // R; the Intra32x32 cu num
    hi_u32                 intra16x16_cu_num; // R; the Intra16x16 cu num
    hi_u32                 intra8x8_cu_num; // R; the Intra8x8   cu num
    hi_u32                 intra4x4_cu_num; // R; the Intra4x4   cu num

    hi_venc_ref_type       ref_type; // R; Type of encoded frames in advanced frame skipping reference mode
    hi_u32                 update_attr_cnt; // R; Number of times that channel attributes or
                                            // parameters (including RC parameters) are set
    hi_u32                 start_qp; // R; the start Qp of encoded frames
    hi_u32                 mean_qp; // R; the mean Qp of encoded frames
    hi_bool                is_p_skip;
} hi_venc_h265_stream_info;

// Defines the features of an jpege stream
typedef struct {
    hi_u32 pic_bytes; // R; the coded picture stream byte number
    hi_u32 update_attr_cnt; // R; Number of times that channel attributes or parameters(including RC parameters) are set
    hi_u32 qfactor; // R; image quality
} hi_venc_jpeg_stream_info;

// Defines the features of an jpege stream
typedef struct {
    hi_u32 pic_bytes;
    hi_u32 update_attr_cnt;
} hi_venc_prores_stream_info;

// the advance information of the h264e
typedef struct {
    hi_u32             residual_bits; // R; the residual num
    hi_u32             head_bits; // R; the head bit num
    hi_u32             madi_val; // R; the madi value
    hi_u32             madp_val; // R; the madp value
    hi_double          psnr_val; // R; the PSNR value
    hi_u32             sse_lcu_cnt; // R; the lcu cnt of the sse
    hi_u64             sse_sum; // R; the sum of the sse
    hi_venc_sse_info   sse_info[HI_VENC_MAX_SSE_NUM]; // R; the information of the sse
    hi_u32             qp_hist[HI_VENC_QP_HIST_NUM]; // R; the Qp histogram value
    hi_u32             move_scene16x16_num; // R; the 16x16 cu num of the move scene
    hi_u32             move_scene_bits; // R; the stream bit num of the move scene
} hi_venc_h264_adv_stream_info;

// the advance information of the h265e
typedef struct {
    hi_u32             residual_bits; // R; the residual num
    hi_u32             head_bits; // R; the head bit num
    hi_u32             madi_val; // R; the madi value
    hi_u32             madp_val; // R; the madp value
    hi_double          psnr_val; // R; the PSNR value
    hi_u32             sse_lcu_cnt; // R; the lcu cnt of the sse
    hi_u64             sse_sum; // R; the sum of the sse
    hi_venc_sse_info   sse_info[HI_VENC_MAX_SSE_NUM]; // R; the information of the sse
    hi_u32             qp_hist[HI_VENC_QP_HIST_NUM]; // R; the Qp histogram value
    hi_u32             move_scene32x32_num; // R; the 32x32 cu num of the move scene
    hi_u32             move_scene_bits; // R; the stream bit num of the move scene
} hi_venc_h265_adv_stream_info;

// Defines the features of an stream
typedef struct {
    hi_venc_pack ATTRIBUTE *pack; // R; stream pack attribute
    hi_u32      ATTRIBUTE pack_cnt; // R; the pack number of one frame stream
    hi_u32      seq; // R; the list number of stream

    union {
        hi_venc_h264_stream_info   h264_info; // R; the stream info of h264
        hi_venc_jpeg_stream_info   jpeg_info; // R; the stream info of jpeg
        hi_venc_h265_stream_info   h265_info; // R; the stream info of h265
        hi_venc_prores_stream_info prores_info; // R; the stream info of prores
    };

    union {
        hi_venc_h264_adv_stream_info   h264_adv_info; // R; the stream info of h264
        hi_venc_h265_adv_stream_info   h265_adv_info; // R; the stream info of h265
    };
} hi_venc_stream;

typedef struct {
    hi_venc_ref_type ref_type; // R;Type of encoded frames in advanced frame skipping reference mode

    hi_u32  pic_bytes; // R;the coded picture stream byte number
    hi_u32  pic_cnt; // R;When channel attributes 'is_by_frame == 1', it means count of frames.
                     // When channel attributes 'is_by_frame == 0', it means count of packets
    hi_u32  start_qp; // R;the start Qp of encoded frames
    hi_u32  mean_qp; // R;the mean Qp of encoded frames
    hi_bool is_p_skip;

    hi_u32  residual_bits; // R;residual
    hi_u32  head_bits; // R;head information
    hi_u32  madi_val; // R;madi
    hi_u32  madp_val; // R;madp
    hi_u64  sse_sum; // R;Sum of SSE value
    hi_u32  sse_lcu_cnt; // R;Sum of LCU number
    hi_double psnr_val; // R;PSNR
} hi_venc_stream_info;

// the status of the venc chnl
typedef struct {
    hi_u32 left_pics; // R; left picture number
    hi_u32 left_stream_bytes; // R; left stream bytes
    hi_u32 left_stream_frames; // R; left stream frames
    hi_u32 cur_packs; // R; pack number of current frame
    hi_u32 left_recv_pics; // R; Number of frames to be received.
    hi_u32 left_enc_pics; // R; Number of frames to be encoded.
    hi_bool is_jpeg_snap_end; // R; the end of Snap.
    hi_u64 release_pic_pts;
    hi_venc_stream_info stream_info;
} hi_venc_chn_status;

// the attribute of h264e
typedef struct {
    hi_bool rcn_ref_share_buf_en; // RW; Range:[0, 1]; Whether to enable the Share Buf of Rcn and Ref
    hi_u32 frame_buf_ratio;
} hi_venc_h264_attr;

// the attribute of h265e
typedef struct {
    hi_bool rcn_ref_share_buf_en; // RW; Range:[0, 1]; Whether to enable the Share Buf of Rcn and Ref
    hi_u32 frame_buf_ratio;
} hi_venc_h265_attr;

// the size of array is 2,that is the maximum
typedef struct {
    hi_u8   large_thumbnail_num; // RW; Range:[0, 2]; the large thumbnail pic num of the MPF
    hi_video_size  large_thumbnail_size[HI_VENC_MAX_MPF_NUM]; // RW; The resolution of large ThumbNail
} hi_venc_mpf_cfg;

typedef enum {
    HI_VENC_PIC_RECV_SINGLE = 0,
    HI_VENC_PIC_RECV_MULTI,

    HI_VENC_PIC_RECV_BUTT
} hi_venc_pic_recv_mode;

// the attribute of jpege
typedef struct {
    hi_bool                  dcf_en; // RW; Range:[0, 1]; support dcf
    hi_venc_mpf_cfg          mpf_cfg; // RW; Range:[0, 1]; config of Mpf
    hi_venc_pic_recv_mode    recv_mode; // RW; Config the receive mode;
} hi_venc_jpeg_attr;

// the frame rate of PRORES
typedef enum {
    HI_VENC_PRORES_FRAME_RATE_UNKNOWN = 0,
    HI_VENC_PRORES_FRAME_RATE_23_976,
    HI_VENC_PRORES_FRAME_RATE_24,
    HI_VENC_PRORES_FRAME_RATE_25,
    HI_VENC_PRORES_FRAME_RATE_29_97,
    HI_VENC_PRORES_FRAME_RATE_30,
    HI_VENC_PRORES_FRAME_RATE_50,
    HI_VENC_PRORES_FRAME_RATE_59_94,
    HI_VENC_PRORES_FRAME_RATE_60,
    HI_VENC_PRORES_FRAME_RATE_100,
    HI_VENC_PRORES_FRAME_RATE_119_88,
    HI_VENC_PRORES_FRAME_RATE_120,
    HI_VENC_PRORES_FRAME_RATE_BUTT
} hi_venc_prores_frame_rate;

// the aspect ratio of PRORES
typedef enum {
    HI_VENC_PRORES_ASPECT_RATIO_UNKNOWN = 0,
    HI_VENC_PRORES_ASPECT_RATIO_SQUARE,
    HI_VENC_PRORES_ASPECT_RATIO_4_3,
    HI_VENC_PRORES_ASPECT_RATIO_16_9,
    HI_VENC_PRORES_ASPECT_RATIO_BUTT
} hi_venc_prores_aspect_ratio;

// the attribute of PRORES
typedef struct {
    hi_char             identifier[HI_VENC_PRORES_MAX_ID_CHAR_NUM];
    hi_venc_prores_frame_rate    frame_rate_code;
    hi_venc_prores_aspect_ratio aspect_ratio;
} hi_venc_prores_attr;

// the attribute of the Venc
typedef struct {
    hi_payload_type  type; // RW; the type of payload

    hi_u32  max_pic_width; // RW; maximum width of a picture to be encoded, in pixel
    hi_u32  max_pic_height; // RW; maximum height of a picture to be encoded, in pixel

    hi_u32  buf_size; // RW; stream buffer size
    hi_u32  profile; // RW; Range:[0, 3]
                        // H.264: 0: baseline, 1:MP, 2:HP, 3: SVC-T [0, 3]
                        // H.265: 0: MP, 1:Main 10  [0 1]
                        // Jpege/MJpege:   0:Baseline
                        // prores: 0:ProRes Proxy; 1:ProRes 422(LT); 2:ProRes 422; 3:ProRes 422(HQ)
    hi_bool is_by_frame; // RW; Range:[0, 1]; get stream mode is slice mode or frame mode
    hi_u32  pic_width; // RW; width of a picture to be encoded, in pixel
    hi_u32  pic_height; // RW; height of a picture to be encoded, in pixel
    union {
        hi_venc_h264_attr h264_attr; // attributes of H264e
        hi_venc_h265_attr h265_attr; // attributes of H265e
        hi_venc_jpeg_attr  jpeg_attr; // attributes of jpeg
        hi_venc_prores_attr prores_attr; // attributes of prores
    };
} hi_venc_attr;

// rc mode
typedef enum {
    HI_VENC_RC_MODE_H264_CBR = 1,
    HI_VENC_RC_MODE_H264_VBR,
    HI_VENC_RC_MODE_H264_AVBR,
    HI_VENC_RC_MODE_H264_QVBR,
    HI_VENC_RC_MODE_H264_CVBR,
    HI_VENC_RC_MODE_H264_FIXQP,
    HI_VENC_RC_MODE_H264_QPMAP,
    HI_VENC_RC_MODE_MJPEG_CBR,
    HI_VENC_RC_MODE_MJPEG_VBR,
    HI_VENC_RC_MODE_MJPEG_FIXQP,
    HI_VENC_RC_MODE_H265_CBR,
    HI_VENC_RC_MODE_H265_VBR,
    HI_VENC_RC_MODE_H265_AVBR,
    HI_VENC_RC_MODE_H265_QVBR,
    HI_VENC_RC_MODE_H265_CVBR,
    HI_VENC_RC_MODE_H265_FIXQP,
    HI_VENC_RC_MODE_H265_QPMAP,
    HI_VENC_RC_MODE_BUTT,
} hi_venc_rc_mode;

// the attribute of h264e cbr
typedef struct {
    hi_u32      gop; // RW; Range:[1, 65536]; the interval of I Frame.
    hi_u32      stats_time; // RW; Range:[1, 60]; the rate statistic time, the unit is senconds(s)
    hi_u32      src_frame_rate; // RW; Range:[1, 240]; the input frame rate of the venc chnnel
    hi_u32      dst_frame_rate; // RW; Range:[1, 240]; the target frame rate of the venc chnnel,
                                  // can not be larger than src_frame_rate
    hi_u32      bit_rate; // RW; Range:[2, 614400]; average bitrate
} hi_venc_h264_cbr;

// the attribute of h264e vbr
typedef struct {
    hi_u32      gop; // RW; Range:[1, 65536]; the interval of ISLICE.
    hi_u32      stats_time; // RW; Range:[1, 60]; the rate statistic time, the unit is senconds(s)
    hi_u32      src_frame_rate; // RW; Range:[1, 240]; the input frame rate of the venc chnnel
    hi_u32      dst_frame_rate; // RW; Range:[1, 240]; the target frame rate of the venc chnnel,
                                  // can not be larger than src_frame_rate
    hi_u32      max_bit_rate; // RW; Range:[2, 614400];the max bitrate
} hi_venc_h264_vbr;

// the attribute of h264e cvbr
typedef struct {
    hi_u32      gop; // RW; Range:[1, 65536]; the interval of ISLICE.
    hi_u32      stats_time; // RW; Range:[1, 60]; the rate statistic time, the unit is senconds(s)
    hi_u32      src_frame_rate; // RW; Range:[1, 240]; the input frame rate of the venc chnnel
    hi_u32      dst_frame_rate; // RW; Range:[1, 240]; the target frame rate of the venc chnnel,
                                  // can not be larger than src_frame_rate
    hi_u32      max_bit_rate; // RW; Range:[2, 614400];the max bitrate

    hi_u32      short_term_stats_time; // RW; Range:[1, 120]; the long-term rate statistic time,
                                      // the unit is second (s)
    hi_u32      long_term_stats_time; // RW; Range:[1, 1440]; the long-term rate statistic time,
                                     // the unit is long_term_stat_time_unit
    hi_u32      long_term_max_bit_rate; // RW; Range:[2, 614400];the long-term target max bitrate,
                                       // can not be larger than max_bit_rate,the unit is kbps
    hi_u32      long_term_min_bit_rate; // RW; Range:[0, 614400];the long-term target min bitrate,
                                       // can not be larger than long_term_max_bitrate,the unit is kbps
} hi_venc_h264_cvbr;

// the attribute of h264e avbr
typedef struct {
    hi_u32      gop; // RW; Range:[1, 65536]; the interval of ISLICE.
    hi_u32      stats_time; // RW; Range:[1, 60]; the rate statistic time, the unit is senconds(s)
    hi_u32      src_frame_rate; // RW; Range:[1, 240]; the input frame rate of the venc chnnel
    hi_u32      dst_frame_rate; // RW; Range:[1, 240]; the target frame rate of the venc chnnel,
                                  // can not be larger than src_frame_rate
    hi_u32      max_bit_rate; // RW; Range:[2, 614400];the max bitrate
} hi_venc_h264_avbr;

// the attribute of h264e qpmap
typedef struct {
    hi_u32      gop; // RW; Range:[1, 65536]; the interval of ISLICE.
    hi_u32      stats_time; // RW; Range:[1, 60]; the rate statistic time, the unit is senconds(s)
    hi_u32      src_frame_rate; // RW; Range:[1, 240]; the input frame rate of the venc chnnel
    hi_u32      dst_frame_rate; // RW; Range:[1, 240]; the target frame rate of the venc chnnel,
                                  // can not be larger than src_frame_rate
} hi_venc_h264_qpmap;

typedef struct {
    hi_u32      gop; // RW; Range:[1, 65536];the interval of ISLICE.
    hi_u32      stats_time; // RW; Range:[1, 60]; the rate statistic time, the unit is senconds(s)
    hi_u32      src_frame_rate; // RW; Range:[1, 240]; the input frame rate of the venc chnnel
    hi_u32      dst_frame_rate; // RW; Range:[1, 240]; the target frame rate of the venc chnnel,
                                  // can not be larger than src_frame_rate
    hi_u32      target_bit_rate; // RW; Range:[2, 614400]; the target bitrate
} hi_venc_h264_qvbr;

// qpmap mode
typedef enum {
    HI_VENC_RC_QPMAP_MODE_MEAN_QP = 0,
    HI_VENC_RC_QPMAP_MODE_MIN_QP,
    HI_VENC_RC_QPMAP_MODE_MAX_QP,

    HI_VENC_RC_QPMAP_MODE_BUTT,
} hi_venc_rc_qpmap_mode;

// the attribute of h265e qpmap
typedef struct {
    hi_u32      gop; // RW; Range:[1, 65536]; the interval of ISLICE.
    hi_u32      stats_time; // RW; Range:[1, 60]; the rate statistic time, the unit is senconds(s)
    hi_u32      src_frame_rate; // RW; Range:[1, 240]; the input frame rate of the venc chnnel
    hi_u32      dst_frame_rate; // RW; Range:[1, 240]; the target frame rate of the venc chnnel,
                                  // can not be larger than src_frame_rate
    hi_venc_rc_qpmap_mode qpmap_mode; // RW;  the QpMap Mode.
} hi_venc_h265_qpmap;

// the attribute of mjpege fixqp
typedef struct {
    hi_u32      src_frame_rate; // RW; Range:[1, 240]; the input frame rate of the venc chnnel
    hi_u32      dst_frame_rate; // RW; Range:[1, 240]; the target frame rate of the venc chnnel,
                                  // can not be larger than src_frame_rate
    hi_u32      qfactor; // RW; Range:[1,99];image quality.
} hi_venc_mjpeg_fixqp;

// the attribute of mjpege cbr
typedef struct {
    hi_u32      stats_time; // RW; Range:[1, 60]; the rate statistic time, the unit is senconds(s)
    hi_u32      src_frame_rate; // RW; Range:[1, 240]; the input frame rate of the venc chnnel
    hi_u32      dst_frame_rate; // RW; Range:[1, 240]; the target frame rate of the venc chnnel,
                                  // can not be larger than src_frame_rate
    hi_u32      bit_rate; // RW; Range:[2, 614400]; average bitrate
} hi_venc_mjpeg_cbr;

// the attribute of mjpege vbr
typedef struct {
    hi_u32      stats_time; // RW; Range:[1, 60]; the rate statistic time, the unit is senconds(s)
    hi_u32      src_frame_rate; // RW; Range:[1, 240]; the input frame rate of the venc chnnel
    hi_u32      dst_frame_rate; // RW; Range:[1, 240]; the target frame rate of the venc chnnel,
                                  // can not be larger than src_frame_rate
    hi_u32      max_bit_rate; // RW; Range:[2, 614400];the max bitrate
} hi_venc_mjpeg_vbr;

// the attribute of h264e fixqp
typedef struct {
    hi_u32      gop; // RW; Range:[1, 65536]; the interval of ISLICE.
    hi_u32      src_frame_rate; // RW; Range:[1, 240]; the input frame rate of the venc chnnel
    hi_u32     dst_frame_rate; // RW; Range:[1, 240]; the target frame rate of the venc chnnel,
                                  // can not be larger than src_frame_rate
    hi_u32      i_qp; // RW; Range:[0, 51]; qp of the i frame
    hi_u32      p_qp; // RW; Range:[0, 51]; qp of the p frame
    hi_u32      b_qp; // RW; Range:[0, 51]; qp of the b frame
} hi_venc_h264_fixqp;

typedef hi_venc_h264_cbr    hi_venc_h265_cbr;
typedef hi_venc_h264_vbr    hi_venc_h265_vbr;
typedef hi_venc_h264_avbr   hi_venc_h265_avbr;
typedef hi_venc_h264_fixqp  hi_venc_h265_fixqp;
typedef hi_venc_h264_qvbr   hi_venc_h265_qvbr;
typedef hi_venc_h264_cvbr   hi_venc_h265_cvbr;

// the attribute of rc
typedef struct {
    hi_venc_rc_mode rc_mode; // RW; the type of rc
    union {
        hi_venc_h264_cbr    h264_cbr;
        hi_venc_h264_vbr    h264_vbr;
        hi_venc_h264_avbr   h264_avbr;
        hi_venc_h264_qvbr   h264_qvbr;
        hi_venc_h264_cvbr   h264_cvbr;
        hi_venc_h264_fixqp  h264_fixqp;
        hi_venc_h264_qpmap  h264_qpmap;

        hi_venc_mjpeg_cbr   mjpeg_cbr;
        hi_venc_mjpeg_vbr   mjpeg_vbr;
        hi_venc_mjpeg_fixqp mjpeg_fixqp;

        hi_venc_h265_cbr    h265_cbr;
        hi_venc_h265_vbr    h265_vbr;
        hi_venc_h265_avbr   h265_avbr;
        hi_venc_h265_qvbr   h265_qvbr;
        hi_venc_h265_cvbr   h265_cvbr;
        hi_venc_h265_fixqp  h265_fixqp;
        hi_venc_h265_qpmap  h265_qpmap;
    };
} hi_venc_rc_attr;

// the gop mode
typedef enum {
    HI_VENC_GOP_MODE_NORMAL_P    = 0, // NORMALP
    HI_VENC_GOP_MODE_DUAL_P      = 1, // DUALP
    HI_VENC_GOP_MODE_SMART_P     = 2, // SMARTP
    HI_VENC_GOP_MODE_ADV_SMART_P  = 3, // ADVSMARTP
    HI_VENC_GOP_MODE_BIPRED_B    = 4, // BIPREDB
    HI_VENC_GOP_MODE_LOW_DELAY_B  = 5, // LOWDELAYB

    HI_VENC_GOP_MODE_BUTT,
} hi_venc_gop_mode;

// the attribute of the normalp
typedef struct {
    hi_s32   ip_qp_delta; // RW; Range:[-10, 30]; QP variance between P frame and I frame
} hi_venc_gop_normal_p;

// the attribute of the dualp
typedef struct {
    hi_u32 sp_interval; // RW; Range:[0, 65536]; Interval of the special P frames,
                          // 1 is not supported and should be less than Gop
    hi_s32 sp_qp_delta; // RW; Range:[-10, 30]; QP variance between P frame and special P frame
    hi_s32 ip_qp_delta; // RW; Range:[-10, 30]; QP variance between P frame and I frame
} hi_venc_gop_dual_p;

// the attribute of the smartp
typedef struct {
    hi_u32  bg_interval; // RW; Interval of the long-term reference frame, can not be less than gop
    hi_s32  bg_qp_delta; // RW; Range:[-10, 30]; QP variance between P frame and Bg frame
    hi_s32  vi_qp_delta; // RW; Range:[-10, 30]; QP variance between P frame and virtual I  frame
} hi_venc_gop_smart_p;

// the attribute of the advsmartp
typedef struct {
    hi_u32  bg_interval; // RW; Interval of the long-term reference frame, can not be less than gop
    hi_s32  bg_qp_delta; // RW; Range:[-10, 30]; QP variance between P frame and Bg frame
    hi_s32  vi_qp_delta; // RW; Range:[-10, 30]; QP variance between P frame and virtual I  frame
} hi_venc_gop_adv_smart_p;

// the attribute of the bipredb
typedef struct {
    hi_u32 b_frame_num; // RW; Range:[1, 3]; Number of B frames
    hi_s32 b_qp_delta; // RW; Range:[-10, 30]; QP variance between P frame and B frame
    hi_s32 ip_qp_delta; // RW; Range:[-10, 30]; QP variance between P frame and I frame
} hi_venc_gop_bipred_b;

// the attribute of the gop
typedef struct {
    hi_venc_gop_mode gop_mode; // RW; Encoding GOP type
    union {
        hi_venc_gop_normal_p   normal_p; // attributes of normal P
        hi_venc_gop_dual_p     dual_p; // attributes of dual   P
        hi_venc_gop_smart_p    smart_p; // attributes of Smart P
        hi_venc_gop_adv_smart_p adv_smart_p; // attributes of AdvSmart P
        hi_venc_gop_bipred_b   bipred_b; // attributes of b
    };

} hi_venc_gop_attr;

// the attribute of the venc chnl
typedef struct {
    hi_venc_attr     venc_attr; // the attribute of video encoder
    hi_venc_rc_attr  rc_attr; // the attribute of rate  ctrl
    hi_venc_gop_attr gop_attr; // the attribute of gop
} hi_venc_chn_attr;

// the param of receive picture
typedef struct {
    hi_s32 recv_pic_num; // RW; Range:[-1, 2147483647]; Number of frames received and
                          // encoded by the encoding channel,0 is not supported
} hi_venc_start_param;

// venc mode type
typedef enum {
    HI_VENC_MOD_VENC = 1,
    HI_VENC_MOD_H264,
    HI_VENC_MOD_H265,
    HI_VENC_MOD_JPEG,
    HI_VENC_MOD_RC,
    HI_VENC_MOD_BUTT
} hi_venc_mod_type;

// the param of the h264e mod
typedef struct {
    hi_u32          one_stream_buf; // RW; Range:[0, 1]; one stream buffer
    hi_u32          mini_buf_mode; // RW; Range:[0, 1]; H264e MiniBufMode
    hi_u32          low_power_mode; // RW; Range:[0, 1]; H264e PowerSaveEn
    hi_vb_src       vb_src; // RW; H264e VBSource
    hi_bool         qp_hist_en; // RW; Range:[0, 1]
    hi_u32          max_user_data_len; // RW; Range:[0, 65536]; one user data buffer len
} hi_venc_h264_mod_param;

// the param of the h265e mod
typedef struct {
    hi_u32          one_stream_buf; // RW; Range:[0, 1]; one stream buffer
    hi_u32          mini_buf_mode; // RW; Range:[0, 1]; H265e MiniBufMode
    hi_u32          low_power_mode; // RW; Range:[0, 2]; H265e PowerSaveEn
    hi_vb_src       vb_src; // RW; H265e VBSource
    hi_bool         qp_hist_en; // RW; Range:[0, 1]
    hi_u32          max_user_data_len; // RW; Range:[0, 65536]; one user data buffer len
} hi_venc_h265_mod_param;

// the param of the jpege mod
typedef struct {
    hi_u32  one_stream_buf; // RW; Range:[0, 1]; one stream buffer
    hi_u32  mini_buf_mode; // RW; Range:[0, 1]; Jpege MiniBufMode
    hi_u32  clear_stream_buf; // RW; Range:[0, 1]; JpegClearStreamBuf
    hi_u32  dering_mode; // RW; Range:[0, 1]; Jpege Dering Mode
} hi_venc_jpeg_mod_param;

// the param of the venc mod
typedef struct {
    hi_u32 buf_cache; // RW; Range:[0, 1]; VencBufferCache
    hi_u32 frame_buf_recycle; // RW; Range:[0, 1]; FrameBufRecycle
} hi_venc_venc_mod_param;

// the param of the mod
typedef struct {
    hi_venc_mod_type mod_type; // RW; VencModType
    union {
        hi_venc_venc_mod_param venc_mod_param;
        hi_venc_h264_mod_param h264_mod_param;
        hi_venc_h265_mod_param h265_mod_param;
        hi_venc_jpeg_mod_param jpeg_mod_param;
    };
} hi_venc_mod_param;

// the param of the jpege
typedef struct {
    hi_u32 qfactor; // RW; Range:{0xFFFFFFFF, [1, 100]}; Qfactor value
    hi_u8  y_qt[HI_VENC_JPEG_QT_COEF_NUM]; // Reserved
    hi_u8  cb_qt[HI_VENC_JPEG_QT_COEF_NUM]; // Reserved
    hi_u8  cr_qt[HI_VENC_JPEG_QT_COEF_NUM]; // Reserved
    hi_u32 mcu_per_ecs; // Reserved
    hi_bool ecs_output_en;
} hi_venc_jpeg_param;

typedef struct {
    hi_u8 dc_bits[16];
    hi_u8 dc_value[12];
} hi_venc_huffman_dc_table;

typedef struct {
    hi_u8 ac_bits[16];
    hi_u8 ac_value[162];
} hi_venc_huffman_ac_table;

// the huffman param of the jpege
typedef struct {
    hi_venc_huffman_dc_table dc_tables[3]; // y, Cb, Cr
    hi_venc_huffman_ac_table ac_tables[3]; // y, Cb, Cr
    hi_u32 reserved[2];
} hi_venc_jpeg_huffman_param;

// the param of the crop
typedef struct {
    hi_bool enable; // RW; Range:[0, 1]; Crop region enable
    hi_rect rect; // RW; Crop region, note: s32X must be multi of 16
} hi_crop_info;

// the param of the venc frame rate
typedef struct {
    hi_s32 src_frame_rate; // RW; Range:[0, 240]; Input frame rate of a channel
    hi_s32 dst_frame_rate; // RW; Range:[0, 240]; Output frame rate of a channel
} hi_frame_rate_ctrl;

// the param of the venc encode chnl
typedef struct {
    hi_bool color_to_grey_en; // RW; Range:[0, 1]; Whether to enable Color2Grey
    hi_u32 priority; // RW; Range:[0, 1]; The priority of the coding chnl
    hi_u32 max_stream_cnt; // RW: Range:[0, 4294967295]; Maximum number of frames in a stream buffer
    hi_u32 poll_wake_up_frame_cnt; // RW: Range:(0, 4294967295]; the frame num needed to wake up  obtaining streams
    hi_crop_info crop_info;
    hi_frame_rate_ctrl frame_rate;
} hi_venc_chn_param;

// the scene mode of the venc encode chnl
typedef enum {
    HI_VENC_SCENE_0 = 0, // RW; A scene in which the camera does not move or periodically moves continuously
    HI_VENC_SCENE_1 = 1, // RW; Motion scene at high bit rate
    HI_VENC_SCENE_2 = 2, // RW; It has regular continuous motion at medium bit rate and
                         // the encoding pressure is relatively large
    HI_VENC_SCENE_BUTT
} hi_venc_scene_mode;

// The param of H264e cbr
typedef struct {
    hi_u32  max_i_proportion; // RW; Range:[1, 100]; the max ratio of i frame and p frame */
                              // can not be smaller than min_i_proportion */
    hi_u32  min_i_proportion; // RW; Range:[1, 100]; the min ratio of i frame and p frame,
    hi_u32  max_qp; // RW; Range:[0, 51];the max QP value */
    hi_u32  min_qp; // RW; Range:[0, 51]; the min QP value,can not be larger than max_qp */
    hi_u32  max_i_qp; // RW; Range:[0, 51]; max qp for i frame */
    hi_u32  min_i_qp; // RW; Range:[0, 51]; min qp for i frame,can not be larger than max_i_qp */
    hi_s32  max_reencode_times; // RW; Range:[0, 3]; Range:max number of re-encode times */
    hi_bool qpmap_en; // RW; Range:[0, 1]; enable qpmap */
} hi_venc_h264_cbr_param;

// The param of H264e vbr
typedef struct {
    hi_s32  chg_pos; // RW; Range:[50, 100]; Indicates the ratio of the current bit rate to
                          // the maximum bit rate when the QP value starts to be adjusted
    hi_u32  max_i_proportion; // RW; Range:[1, 100] ; the max ratio of i frame and p frame
                              // can not be smaller than min_i_proportion
    hi_u32  min_i_proportion; // RW; Range:[1, 100] ; the min ratio of i frame and p frame,
    hi_s32  max_reencode_times; // RW; Range:[0, 3]; max number of re-encode times
    hi_bool qpmap_en; // RW; Range:[0, 1]; enable qpmap

    hi_u32  max_qp; // RW; Range:[0, 51]; the max P B qp
    hi_u32  min_qp; // RW; Range:[0, 51]; the min P B qp,can not be larger than max_qp
    hi_u32  max_i_qp; // RW; Range:[0, 51]; the max I qp
    hi_u32  min_i_qp; // RW; Range:[0, 51]; the min I qp ,can not be larger than max_i_qp
} hi_venc_h264_vbr_param;

// The param of H264e avbr
typedef struct {
    hi_s32  chg_pos; // RW; Range:[50, 100]; Indicates the ratio of the current bit rate to
                          // the maximum bit rate when the QP value starts to be adjusted
    hi_u32  max_i_proportion;  // RW; Range:[1, 100] ; the max ratio of i frame and p frame,
                               // can not be smaller than min_i_proportion
    hi_u32  min_i_proportion;  // RW; Range:[1, 100] ; the min ratio of i frame and p frame
    hi_s32  max_reencode_times; // RW; Range:[0, 3]; max number of re-encode times
    hi_bool qpmap_en; // RW; Range:[0, 1]; enable qpmap

    hi_s32  min_still_percent; // RW; Range:[5, 100]; the min percent of target bitrate for still scene
    hi_u32  max_still_qp; // RW; Range:[0, 51]; the max QP value of I frame for still scene,can not be
                           // smaller than min_i_qp and can not be larger than max_i_qp
    hi_u32  min_still_psnr; // RW; reserved,Invalid member currently

    hi_u32  max_qp; // RW; Range:[0, 51]; the max P B qp
    hi_u32  min_qp; // RW; Range:[0, 51]; the min P B qp,can not be larger than max_qp
    hi_u32  max_i_qp; // RW; Range:[0, 51]; the max I qp
    hi_u32  min_i_qp; // RW; Range:[0, 51]; the min I qp,can not be larger than max_i_qp
    hi_u32  min_qp_delta; // RW; Range:[0, 4];Difference between FrameLevelMinQp and min_qp,
                           // FrameLevelMinQp = MinQp(or MinIQp) + MinQpDelta
    hi_u32  motion_sensitivity; // RW; Range:[0, 100]; Motion Sensitivity
    hi_bool save_bitrate_en;
} hi_venc_h264_avbr_param;

// The param of H264e qvbr
typedef struct {
    hi_u32  max_i_proportion; // RW; Range:[1, 100]; the max ratio of i frame and p frame,
                              // can not be smaller than min_i_proportion
    hi_u32  min_i_proportion; // RW; Range:[1, 100] ;the min ratio of i frame and p frame
    hi_s32  max_reencode_times; // RW; Range:[0, 3];max number of re-encode times [0, 3]
    hi_bool qpmap_en; // RW; Range:[0, 1]; enable qpmap

    hi_u32  max_qp; // RW; Range:[0, 51]; the max P B qp
    hi_u32  min_qp; // RW; Range:[0, 51]; the min P B qp,can not be larger than max_qp
    hi_u32  max_i_qp; // RW; Range:[0, 51]; the max I qp
    hi_u32  min_i_qp; // RW; Range:[0, 51]; the min I qp,can not be larger than max_i_qp

    hi_s32  max_bit_percent; // RW; Range:[30, 180]; Indicate the ratio of bitrate  upper limit
    hi_s32  min_bit_percent; // RW; Range:[30, 180]; Indicate the ratio of bitrate  lower limit,
                             // can not be larger than bit_percent_ul
    hi_s32  max_psnr_fluctuate; // RW; Range:[18, 40]; Reduce the target bitrate
                                // when the value of psnr approch the upper limit
    hi_s32  min_psnr_fluctuate; // RW; Range:[18, 40]; Increase the target bitrate when the value of psnr
                                // approch the lower limit, can not be larger than psnr_fluctuate_ul
} hi_venc_h264_qvbr_param;

// The param of H264e cvbr
typedef struct {
    hi_u32  max_i_proportion; // RW; Range:[1, 100] ; the max ratio of i frame and p frame,
                              // can not be smaller than min_i_proportion
    hi_u32  min_i_proportion; // RW; Range:[1, 100] ; the min ratio of i frame and p frame
    hi_s32  max_reencode_times; // RW; Range:[0, 3]; max number of re-encode times
    hi_bool qpmap_en; // RW; Range:[0, 1]; enable qpmap

    hi_u32  max_qp; // RW; Range:[0, 51]; the max P B qp
    hi_u32  min_qp; // RW; Range:[0, 51]; the min P B qp,can not be larger than max_qp
    hi_u32  max_i_qp; // RW; Range:[0, 51]; the max I qp
    hi_u32  min_i_qp; // RW; Range:[0, 51]; the min I qp,can not be larger than max_i_qp

    hi_u32  min_qp_delta; // RW; Range:[0, 4];Difference between FrameLevelMinQp and min_qp,
                           // FrameLevelMinQp = MinQp(or MinIQp) + MinQpDelta
    hi_u32  max_qp_delta; // RW; Range:[0, 4];Difference between FrameLevelMaxQp and max_qp,
                           // FrameLevelMaxQp = MaxQp(or MaxIQp) - MaxQpDelta

    hi_u32  extra_bit_percent; // RW; Range:[0, 1000];the extra bits that can be allocated
                                // when the actual bitrate goes above the target bitrate
    hi_u32  long_term_stats_time_unit; // RW; Range:[1, 1800]; the time unit of long_term_stat_time,
                                     // the unit is senconds(s)
    hi_bool save_bitrate_en;
} hi_venc_h264_cvbr_param;

typedef struct {
    hi_u32 max_qfactor;
    hi_u32 min_qfactor;
} hi_venc_mjpeg_cbr_param;

typedef struct {
    hi_s32 chg_pos;
    hi_u32 max_qfactor;
    hi_u32 min_qfactor;
} hi_venc_mjpeg_vbr_param;
// The param of h265e cbr
typedef struct {
    hi_u32  max_i_proportion; // RW; Range:[1, 100]; the max ratio of i frame and p frame,
                              // can not be smaller than min_i_proportion
    hi_u32  min_i_proportion; // RW; Range:[1, 100];the min ratio of i frame and p frame
    hi_u32  max_qp; // RW; Range:[0, 51];the max QP value
    hi_u32  min_qp; // RW; Range:[0, 51];the min QP value ,can not be larger than max_qp
    hi_u32  max_i_qp; // RW; Range:[0, 51];max qp for i frame
    hi_u32  min_i_qp; // RW; Range:[0, 51];min qp for i frame,can not be larger than max_i_qp
    hi_s32  max_reencode_times;  // RW; Range:[0, 3]; Range:max number of re-encode times
    hi_bool qpmap_en; // RW; Range:[0, 1]; enable qpmap
    hi_venc_rc_qpmap_mode qpmap_mode;  // RW; Qpmap Mode
} hi_venc_h265_cbr_param;

// The param of h265e vbr
typedef struct {
    hi_s32  chg_pos; // RW; Range:[50, 100];Indicates the ratio of the current bit rate
                          // to the maximum bit rate when the QP value starts to be adjusted
    hi_u32  max_i_proportion; // RW; [1, 100]the max ratio of i frame and p frame,
                              // can not be smaller than min_i_proportion
    hi_u32  min_i_proportion; // RW; [1, 100]the min ratio of i frame and p frame
    hi_s32  max_reencode_times; // RW; Range:[0, 3]; Range:max number of re-encode times

    hi_u32  max_qp; // RW; Range:[0, 51]; the max P B qp
    hi_u32  min_qp; // RW; Range:[0, 51]; the min P B qp ,can not be larger than max_qp
    hi_u32  max_i_qp; // RW; Range:[0, 51]; the max I qp
    hi_u32  min_i_qp; // RW; Range:[0, 51]; the min I qp,can not be larger than max_i_qp

    hi_bool qpmap_en; // RW; Range:[0, 1]; enable qpmap
    hi_venc_rc_qpmap_mode qpmap_mode; // RW; Qpmap Mode
} hi_venc_h265_vbr_param;

// The param of h265e avbr
typedef struct {
    hi_s32  chg_pos; // RW; Range:[50, 100];Indicates the ratio of the current bit rate to
                          // the maximum bit rate when the QP value starts to be adjusted
    hi_u32  max_i_proportion; // RW; [1, 100]the max ratio of i frame and p frame,
                              // can not be smaller than min_i_proportion
    hi_u32  min_i_proportion; // RW; [1, 100]the min ratio of i frame and p frame
    hi_s32  max_reencode_times; // RW; Range:[0, 3]; Range:max number of re-encode times

    hi_s32  min_still_percent; // RW; Range:[5, 100]; the min percent of target bitrate for still scene
    hi_u32  max_still_qp; // RW; Range:[0, 51]; the max QP value of I frame for still scene,can not be
                           // smaller than u32MinIQp and can not be larger than max_iprop
    hi_u32  min_still_psnr; // RW; reserved

    hi_u32  max_qp; // RW; Range:[0, 51];the max P B qp
    hi_u32  min_qp; // RW; Range:[0, 51];the min P B qp,can not be larger than max_qp
    hi_u32  max_i_qp; // RW; Range:[0, 51];the max I qp
    hi_u32  min_i_qp; // RW; Range:[0, 51];the min I qp,can not be larger than max_i_qp

    hi_u32  min_qp_delta; // RW; Range:[0, 4];Difference between FrameLevelMinQp and min_qp,
                           // FrameLevelMinQp = MinQp(or MinIQp) + MinQpDelta
    hi_u32  motion_sensitivity; // RW; Range:[0, 100]; Motion Sensitivity

    hi_bool qpmap_en; // RW; Range:[0, 1]; enable qpmap
    hi_venc_rc_qpmap_mode qpmap_mode; // RW; Qpmap Mode
} hi_venc_h265_avbr_param;

// The param of h265e qvbr
typedef struct {
    hi_u32  max_i_proportion; // RW; [1, 100];the max ratio of i frame and p frame,
                              // can not be smaller than min_i_proportion
    hi_u32  min_i_proportion; // RW; [1, 100];the min ratio of i frame and p frame
    hi_s32  max_reencode_times; // RW; [0, 3]; max number of re-encode times [0, 3]

    hi_bool qpmap_en; // RW; Range:[0, 1]; enable qpmap
    hi_venc_rc_qpmap_mode qpmap_mode; // RW; Qpmap Mode

    hi_u32  max_qp; // RW; Range:[0, 51]; the max P B qp
    hi_u32  min_qp; // RW; Range:[0, 51]; the min P B qp,can not be larger than max_qp
    hi_u32  max_i_qp; // RW; Range:[0, 51]; the max I qp
    hi_u32  min_i_qp; // RW; Range:[0, 51]; the min I qp,can not be larger than max_i_qp

    hi_s32  max_bit_percent; // RW; Range:[30, 180]; Indicate the ratio of bitrate  upper limit
    hi_s32  min_bit_percent; // RW; Range:[30, 180]; Indicate the ratio of bitrate  lower limit,
                             // can not be larger than bit_percent_ul
    hi_s32  max_psnr_fluctuate; // RW; Range:[18, 40];  Reduce the target bitrate
                               // when the value of psnr approch the upper limit
    hi_s32  min_psnr_fluctuate;  // RW; Range:[18, 40];  Increase the target bitrate when the value of psnr
                                 // approch the lower limit,can not be larger than psnr_fluctuate_ul
} hi_venc_h265_qvbr_param;

// The param of h265e cvbr
typedef struct {
    hi_u32  max_i_proportion; // RW; Range:[1, 100] ; the max ratio of i frame and p frame,
                              // can not be smaller than min_i_proportion
    hi_u32  min_i_proportion; // RW; Range:[1, 100] ; the min ratio of i frame and p frame
    hi_s32  max_reencode_times; // RW; Range:[0, 3]; max number of re-encode times
    hi_bool qpmap_en; // RW; Range:[0, 1]; enable qpmap
    hi_venc_rc_qpmap_mode qpmap_mode; // RW; Qpmap Mode

    hi_u32  max_qp; // RW; Range:[0, 51]; the max P B qp
    hi_u32  min_qp; // RW; Range:[0, 51]; the min P B qp,can not be larger than max_qp
    hi_u32  max_i_qp; // RW; Range:[0, 51]; the max I qp
    hi_u32  min_i_qp; // RW; Range:[0, 51]; the min I qp,can not be larger than max_i_qp

    hi_u32  min_qp_delta; // RW; Range:[0, 4];Difference between FrameLevelMinQp and min_qp,
                           // FrameLevelMinQp = MinQp(or MinIQp) + MinQpDelta
    hi_u32  max_qp_delta; // RW; Range:[0, 4];Difference between FrameLevelMaxQp and max_qp,
                           // FrameLevelMaxQp = MaxQp(or MaxIQp) - MaxQpDelta

    hi_u32  extra_bit_percent; // RW; Range:[0, 1000];the extra ratio of bitrate that can be allocated
                                // when the actual bitrate goes above the long-term target bitrate
    hi_u32  long_term_stats_time_unit; // RW; Range:[1, 1800]; the time unit of LongTermStatTime,
                                     // the unit is senconds(s)
} hi_venc_h265_cvbr_param;

// The param of scene change detect
typedef struct {
    hi_bool detect_scene_chg_en;  // RW; Range:[0, 1]; enable detect scene change.
    hi_bool adapt_insert_idr_frame_en;  // RW; Range:[0, 1]; enable a daptive insertIDR frame.
} hi_venc_scene_chg_detect;

// The param of rc
typedef struct {
    hi_u32 threshold_i[HI_VENC_TEXTURE_THRESHOLD_SIZE]; // RW; Range:[0, 255]; Mad threshold for controlling
                                          // the macroblock-level bit rate of I frames
    hi_u32 threshold_p[HI_VENC_TEXTURE_THRESHOLD_SIZE]; // RW; Range:[0, 255]; Mad threshold for controlling
                                          // the macroblock-level bit rate of P frames
    hi_u32 threshold_b[HI_VENC_TEXTURE_THRESHOLD_SIZE]; // RW; Range:[0, 255]; Mad threshold for controlling
                                         // the macroblock-level bit rate of B frames
    hi_u32 direction; // RW; Range:[0, 16]; The direction for controlling
                             // the macroblock-level bit rate
    hi_u32 row_qp_delta; // RW; Range:[0, 10];the start QP value of each macroblock row
                         // relative to the start QP value
    hi_s32 first_frame_start_qp; // RW; Range:[-1, 51];Start QP value of the first frame
    hi_venc_scene_chg_detect scene_chg_detect;
    union {
        hi_venc_h264_cbr_param  h264_cbr_param;
        hi_venc_h264_vbr_param  h264_vbr_param;
        hi_venc_h264_avbr_param h264_avbr_param;
        hi_venc_h264_qvbr_param h264_qvbr_param;
        hi_venc_h264_cvbr_param h264_cvbr_param;
        hi_venc_h265_cbr_param  h265_cbr_param;
        hi_venc_h265_vbr_param  h265_vbr_param;
        hi_venc_h265_avbr_param h265_avbr_param;
        hi_venc_h265_qvbr_param h265_qvbr_param;
        hi_venc_h265_cvbr_param h265_cvbr_param;
        hi_venc_mjpeg_cbr_param mjpeg_cbr_param;
        hi_venc_mjpeg_vbr_param mjpeg_vbr_param;
    };
} hi_venc_rc_param;

/*
 * @brief create video encoder channel
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] attr: pointer of video encoder channel attribute
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_create_chn(hi_venc_chn chn, const hi_venc_chn_attr *attr);

/*
 * @brief destroy video encoder channel
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_destroy_chn(hi_venc_chn chn);

/*
 * @brief video encoder channel start receive frame
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] recv_param: pointer of receive picture parameter struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_start_chn(hi_venc_chn chn, const hi_venc_start_param *recv_param);

/*
 * @brief video encoder channel stop receive frame
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_stop_chn(hi_venc_chn chn);

/*
 * @brief query video encoder channel status
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [out] status: pointer of video encoder channel status struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_query_status(hi_venc_chn chn, hi_venc_chn_status *status);

/*
 * @brief get stream from video encoder channel
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] milli_sec: -1 is block,0 is no block,other positive number is timeout
 * @param [out] stream: pointer of stream info struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_get_stream(hi_venc_chn chn, hi_venc_stream *stream, hi_s32 milli_sec);

/*
 * @brief release frame from video encoder channel
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] stream: pointer of stream info struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_release_stream(hi_venc_chn chn, hi_venc_stream *stream);

/*
 * @brief send frame to video encoder channel
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] frame: pointer of frame struct
 * @param [in] milli_sec: -1 is block,0 is no block,other positive number is timeout
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_send_frame(hi_venc_chn chn, const hi_video_frame_info *frame, hi_s32 milli_sec);

/*
 * @brief send jpege frame to video encoder channel when the output address of stream is assigned
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] frame: pointer of frame struct
 * @param [in] jpege_stream: pointer of img stream to assign the output address of stream
 * @param [in] milli_sec: -1 is block,0 is no block,other positive number is timeout
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_send_jpege_frame(hi_venc_chn chn, const hi_video_frame_info *frame,
    const hi_img_stream* jpege_stream, hi_s32 milli_sec);

/*
 * @brief send jpege frame to video encoder channel when the output address of stream is assigned
 * @param [in] frame: pointer of frame struct
 * @param [in] jpeg_param: pointer of jpeg encoder quality parameter struct
 * @param [out] size: the predicted size of output buffer of img stream
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_get_jpege_predicted_size(const hi_video_frame_info* frame, const hi_venc_jpeg_param* jpeg_param,
    hi_u32 *size);

/*
 * @brief get video encoder channel's device file handle
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @return success: return positive number
 *         fail: return negative number
 */
hi_s32 hi_mpi_venc_get_fd(hi_venc_chn chn);

/*
 * @brief close video encoder channel's device file handle
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_close_fd(hi_venc_chn chn);

/*
 * @brief set channel's parameters
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] chn_param: pointer of video encoder channel patameter struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_set_chn_param(hi_venc_chn chn, const hi_venc_chn_param *chn_param);

/*
 * @brief set video encoder module parameter
 * @param [in] mod_param: pointer of video encoder module parameter struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_set_mod_param(const hi_venc_mod_param *mod_param);

/*
 * @brief get video encoder module parameter
 * @param [out] mod_param: pointer of video encoder module parameter struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_get_mod_param(hi_venc_mod_param *mod_param);

/*
 * @brief set jpeg image quality
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] jpeg_param: pointer of jpeg encoder quality parameter struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_set_jpeg_param(hi_venc_chn chn, const hi_venc_jpeg_param *jpeg_param);

/*
 * @brief get jpeg image quality
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [out] jpeg_param: pointer of jpeg encoder quality parameter struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_get_jpeg_param(hi_venc_chn chn, hi_venc_jpeg_param *jpeg_param);

/*
 * @brief set jpege huffman parameters
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] jpeg_huffman_param: pointer of jpeg encoder huffman table
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_set_jpeg_huffman_param(hi_venc_chn chn, const hi_venc_jpeg_huffman_param *jpeg_huffman_param);

/*
 * @brief get jpege huffman parameter
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [out] jpeg_huffman_data: pointer of jpeg encoder huffman table
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_get_jpeg_huffman_param(hi_venc_chn chn, hi_venc_jpeg_huffman_param *jpeg_huffman_data);

/*
 * @brief compact huffman and qt tables
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] table_type: compact table type only support 0
 * @param [in] enable: enable compact or not [HI_FALSE, HI_TRUE]
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_compact_jpeg_tables(hi_venc_chn chn, hi_u32 table_type, hi_bool enable);

/*
 * @brief request video encoder channel to encode IDR frame
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] instant: whether to encode IDR frame immediately
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_request_idr(hi_venc_chn chn, hi_bool instant);

/*
 * @brief set encoding scene mode
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] scene_mode: scene mode
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_set_scene_mode(hi_venc_chn chn, const hi_venc_scene_mode scene_mode);

/*
 * @brief set rc parameter
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] rc_param: pointer of video encoder channel patameter struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_set_rc_param(hi_venc_chn chn, const hi_venc_rc_param *rc_param);

/*
 * @brief get rc parameter
 * @param [in] chn: video encoder channel id [0, VENC_MAX_CHN_NUM)
 * @param [in] rc_param: pointer of video encoder channel patameter struct
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_venc_get_rc_param(hi_venc_chn chn, hi_venc_rc_param *rc_param);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif // #ifndef HI_VENC_H_
