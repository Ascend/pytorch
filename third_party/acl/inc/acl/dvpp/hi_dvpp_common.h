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

#ifndef HI_DVPP_COMMON_H_
#define HI_DVPP_COMMON_H_

#include "hi_dvpp_type.h"
#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#define ALIGN_NUM 8
#define DEFAULT_ALIGN 32U
#define ATTRIBUTE __attribute__((aligned(ALIGN_NUM)))

#define ALIGN_UP(x, a) ((((x) + ((a) - 1U)) / (a)) * (a))

#define HI_MAX_COLOR_COMPONENT 3
#define HI_MAX_USER_DATA_NUM 2

#define HI_ERR_APP_ID (0x80000000U + 0x20000000U)

#define HI_DEFINE_ERR(mod, level, err_id) \
    ((hi_s32)((HI_ERR_APP_ID) | ((hi_u32)(mod) << 16) | ((hi_u32)(level) << 13) | ((hi_u32)(err_id))))

typedef enum {
    HI_ERR_INVALID_DEV_ID = 1, // invlalid device ID
    HI_ERR_INVALID_CHN_ID = 2, // invlalid channel ID
    HI_ERR_ILLEGAL_PARAM = 3, // at lease one parameter is illagal
                              // eg, an illegal enumeration value
    HI_ERR_EXIST         = 4, // resource exists
    HI_ERR_UNEXIST       = 5, // resource unexists

    HI_ERR_NULL_PTR      = 6, // using a NULL point

    HI_ERR_NOT_CFG    = 7, // try to enable or initialize system, device
                           // or channel, before configing attribute

    HI_ERR_NOT_SUPPORT   = 8, // operation or type is not supported by NOW
    HI_ERR_NOT_PERM      = 9, // operation is not permitted
                              // eg, try to change static attribute
    HI_ERR_INVALID_PIPE_ID = 10, // invlalid pipe ID
    HI_ERR_INVALID_GRP_ID  = 11, // invlalid stitch group ID

    HI_ERR_NO_MEM         = 12, // failure caused by malloc memory
    HI_ERR_NO_BUF         = 13, // failure caused by malloc buffer

    HI_ERR_BUF_EMPTY     = 14, // no data in buffer
    HI_ERR_BUF_FULL      = 15, // no buffer for new data

    HI_ERR_NOT_READY  = 16, // System is not ready,maybe not initialed or
                            // loaded. Returning the error code when opening
                            // a device file failed.

    HI_ERR_BAD_ADDR       = 17, // bad address,
                                // eg. used for copy_from_user & copy_to_user

    HI_ERR_BUSY          = 18, // resource is busy,
                               // eg. destroy a venc chn without unregister it
    HI_ERR_SIZE_NOT_ENOUGH = 19, // buffer size is smaller than the actual size required
    HI_ERR_TIMEOUT       = 20, // hardware or software timeout
    HI_ERR_SYS_ERROR     = 21, // Internal system error

    HI_ERR_BUTT          = 63, // maxium code, private error code of all modules
                               // must be greater than it
} hi_errno;

typedef enum {
    HI_ERR_LEVEL_DEBUG = 0,  // debug-level
    HI_ERR_LEVEL_INFO,       // informational
    HI_ERR_LEVEL_NOTICE,     // normal but significant condition
    HI_ERR_LEVEL_WARNING,    // warning conditions
    HI_ERR_LEVEL_ERROR,      // error conditions
    HI_ERR_LEVEL_CRIT,       // critical conditions
    HI_ERR_LEVEL_ALERT,      // action must be taken immediately
    HI_ERR_LEVEL_FATAL,      // just for compatibility with previous version
    HI_ERR_LEVEL_BUTT
} hi_err_level;

typedef enum {
    HI_ID_CMPI = 0,
    HI_ID_VB = 1,
    HI_ID_SYS = 2,
    HI_ID_RGN = 3,
    HI_ID_CHNL = 4,
    HI_ID_VDEC = 5,
    HI_ID_AVS = 6,
    HI_ID_VPC = 7,
    HI_ID_VENC = 8,
    HI_ID_SVP = 9,
    HI_ID_H264E = 10,
    HI_ID_JPEGE = 11,
    HI_ID_MPEG4E = 12,
    HI_ID_H265E = 13,
    HI_ID_JPEGD = 14,
    HI_ID_VO = 15,
    HI_ID_VI = 16,
    HI_ID_DIS = 17,
    HI_ID_VALG = 18,
    HI_ID_RC = 19,
    HI_ID_AIO = 20,
    HI_ID_AI = 21,
    HI_ID_AO = 22,
    HI_ID_AENC = 23,
    HI_ID_ADEC = 24,
    HI_ID_VPU = 25,
    HI_ID_PCIV = 26,
    HI_ID_PCIVFMW = 27,
    HI_ID_ISP = 28,
    HI_ID_IVE = 29,
    HI_ID_USER = 30,
    HI_ID_DCCM = 31,
    HI_ID_DCCS = 32,
    HI_ID_PROC = 33,
    HI_ID_LOG = 34,
    HI_ID_VFMW = 35,
    HI_ID_H264D = 36,
    HI_ID_GDC = 37,
    HI_ID_PHOTO = 38,
    HI_ID_FB = 39,
    HI_ID_HDMI = 40,
    HI_ID_VOIE = 41,
    HI_ID_TDE = 42,
    HI_ID_HDR = 43,
    HI_ID_PRORES = 44,
    HI_ID_VGS = 45,

    HI_ID_FD = 47,
    HI_ID_ODT = 48, // Object detection trace
    HI_ID_VQA = 49, // Video quality analysis
    HI_ID_LPR = 50, // Object detection trace
    HI_ID_SVP_NNIE = 51,
    HI_ID_SVP_DSP = 52,
    HI_ID_DPU_RECT = 53,
    HI_ID_DPU_MATCH = 54,

    HI_ID_MOTIONSENSOR = 55,
    HI_ID_MOTIONFUSION = 56,

    HI_ID_GYRODIS = 57,
    HI_ID_PM = 58,
    HI_ID_SVP_ALG = 59,
    HI_ID_IVP = 60,
    HI_ID_MCF = 61,
    HI_ID_VPSS = 62,
    HI_ID_DRV_VPC = 63,
    HI_ID_PNGD = 64,

    HI_ID_VDEC_ADAPT   = 65,
    HI_ID_DCC          = 66,
    HI_ID_VDEC_SERVER  = 67,

    HI_ID_BUTT = 0x100,
} hi_mod_id;

// We just coyp this value of payload type from RTP/RTSP definition
typedef enum {
    HI_PT_PCMU          = 0,
    HI_PT_1016          = 1,
    HI_PT_G721          = 2,
    HI_PT_GSM           = 3,
    HI_PT_G723          = 4,
    HI_PT_DVI4_8K       = 5,
    HI_PT_DVI4_16K      = 6,
    HI_PT_LPC           = 7,
    HI_PT_PCMA          = 8,
    HI_PT_G722          = 9,
    HI_PT_S16BE_STEREO  = 10,
    HI_PT_S16BE_MONO    = 11,
    HI_PT_QCELP         = 12,
    HI_PT_CN            = 13,
    HI_PT_MPEGAUDIO     = 14,
    HI_PT_G728          = 15,
    HI_PT_DVI4_3        = 16,
    HI_PT_DVI4_4        = 17,
    HI_PT_G729          = 18,
    HI_PT_G711A         = 19,
    HI_PT_G711U         = 20,
    HI_PT_G726          = 21,
    HI_PT_G729A         = 22,
    HI_PT_LPCM          = 23,
    HI_PT_CelB          = 25,
    HI_PT_JPEG          = 26,
    HI_PT_CUSM          = 27,
    HI_PT_NV            = 28,
    HI_PT_PICW          = 29,
    HI_PT_CPV           = 30,
    HI_PT_H261          = 31,
    HI_PT_MPEGVIDEO     = 32,
    HI_PT_MPEG2TS       = 33,
    HI_PT_H263          = 34,
    HI_PT_SPEG          = 35,
    HI_PT_MPEG2VIDEO    = 36,
    HI_PT_AAC           = 37,
    HI_PT_WMA9STD       = 38,
    HI_PT_HEAAC         = 39,
    HI_PT_PCM_VOICE     = 40,
    HI_PT_PCM_AUDIO     = 41,
    HI_PT_MP3           = 43,
    HI_PT_ADPCMA        = 49,
    HI_PT_AEC           = 50,
    HI_PT_X_LD          = 95,
    HI_PT_H264          = 96,
    HI_PT_D_GSM_HR      = 200,
    HI_PT_D_GSM_EFR     = 201,
    HI_PT_D_L8          = 202,
    HI_PT_D_RED         = 203,
    HI_PT_D_VDVI        = 204,
    HI_PT_D_BT656       = 220,
    HI_PT_D_H263_1998   = 221,
    HI_PT_D_MP1S        = 222,
    HI_PT_D_MP2P        = 223,
    HI_PT_D_BMPEG       = 224,
    HI_PT_MP4VIDEO      = 230,
    HI_PT_MP4AUDIO      = 237,
    HI_PT_VC1           = 238,
    HI_PT_JVC_ASF       = 255,
    HI_PT_D_AVI         = 256,
    HI_PT_DIVX3         = 257,
    HI_PT_AVS           = 258,
    HI_PT_REAL8         = 259,
    HI_PT_REAL9         = 260,
    HI_PT_VP6           = 261,
    HI_PT_VP6F          = 262,
    HI_PT_VP6A          = 263,
    HI_PT_SORENSON      = 264,
    HI_PT_H265          = 265,
    HI_PT_VP8           = 266,
    HI_PT_MVC           = 267,
    HI_PT_PNG           = 268,
    // add by hisilicon
    HI_PT_AMR           = 1001,
    HI_PT_MJPEG         = 1002,
    HI_PT_AMRWB         = 1003,
    HI_PT_PRORES        = 1006,
    HI_PT_OPUS          = 1007,
    HI_PT_VPC           = 2000,
    HI_PT_BUTT
} hi_payload_type;

typedef enum  {
    HI_DATA_BIT_WIDTH_8 = 0,
    HI_DATA_BIT_WIDTH_10,
    HI_DATA_BIT_WIDTH_12,
    HI_DATA_BIT_WIDTH_14,
    HI_DATA_BIT_WIDTH_16,
    HI_DATA_BIT_WIDTH_BUTT
} hi_data_bit_width;

// we ONLY define picture format used, all unused will be deleted!
typedef enum {
    HI_PIXEL_FORMAT_YUV_400 = 0,
    HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1,
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420 = 2,
    HI_PIXEL_FORMAT_YUV_SEMIPLANAR_422 = 3,
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_422 = 4,
    HI_PIXEL_FORMAT_YUV_SEMIPLANAR_444 = 5,
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_444 = 6,
    HI_PIXEL_FORMAT_YUYV_PACKED_422 = 7,
    HI_PIXEL_FORMAT_UYVY_PACKED_422 = 8,
    HI_PIXEL_FORMAT_YVYU_PACKED_422 = 9,
    HI_PIXEL_FORMAT_VYUY_PACKED_422 = 10,
    HI_PIXEL_FORMAT_YUV_PACKED_444 = 11,
    HI_PIXEL_FORMAT_RGB_888 = 12,
    HI_PIXEL_FORMAT_BGR_888 = 13,
    HI_PIXEL_FORMAT_ARGB_8888 = 14,
    HI_PIXEL_FORMAT_ABGR_8888 = 15,
    HI_PIXEL_FORMAT_RGBA_8888 = 16,
    HI_PIXEL_FORMAT_BGRA_8888 = 17,
    HI_PIXEL_FORMAT_YUV_SEMI_PLANNER_420_10BIT = 18,
    HI_PIXEL_FORMAT_YVU_SEMI_PLANNER_420_10BIT = 19,
    HI_PIXEL_FORMAT_YVU_PLANAR_420 = 20,
    HI_PIXEL_FORMAT_YVU_PLANAR_422 = 21,
    HI_PIXEL_FORMAT_YVU_PLANAR_444 = 22,
    HI_PIXEL_FORMAT_RGB_444 = 23,
    HI_PIXEL_FORMAT_BGR_444 = 24,
    HI_PIXEL_FORMAT_ARGB_4444 = 25,
    HI_PIXEL_FORMAT_ABGR_4444 = 26,
    HI_PIXEL_FORMAT_RGBA_4444 = 27,
    HI_PIXEL_FORMAT_BGRA_4444 = 28,
    HI_PIXEL_FORMAT_RGB_555 = 29,
    HI_PIXEL_FORMAT_BGR_555 = 30,
    HI_PIXEL_FORMAT_RGB_565 = 31,
    HI_PIXEL_FORMAT_BGR_565 = 32,
    HI_PIXEL_FORMAT_ARGB_1555 = 33,
    HI_PIXEL_FORMAT_ABGR_1555 = 34,
    HI_PIXEL_FORMAT_RGBA_1555 = 35,
    HI_PIXEL_FORMAT_BGRA_1555 = 36,
    HI_PIXEL_FORMAT_ARGB_8565 = 37,
    HI_PIXEL_FORMAT_ABGR_8565 = 38,
    HI_PIXEL_FORMAT_RGBA_8565 = 39,
    HI_PIXEL_FORMAT_BGRA_8565 = 40,
    HI_PIXEL_FORMAT_ARGB_CLUT2 = 41,
    HI_PIXEL_FORMAT_ARGB_CLUT4 = 42,

    HI_PIXEL_FORMAT_RGB_BAYER_8BPP = 50,
    HI_PIXEL_FORMAT_RGB_BAYER_10BPP = 51,
    HI_PIXEL_FORMAT_RGB_BAYER_12BPP = 52,
    HI_PIXEL_FORMAT_RGB_BAYER_14BPP = 53,
    HI_PIXEL_FORMAT_RGB_BAYER_16BPP = 54,
    HI_PIXEL_FORMAT_YUV_PLANAR_420 = 55,
    HI_PIXEL_FORMAT_YUV_PLANAR_422 = 56,
    HI_PIXEL_FORMAT_YUV_PLANAR_444 = 57,
    HI_PIXEL_FORMAT_YVU_PACKED_444 = 58,
    HI_PIXEL_FORMAT_XYUV_PACKED_444 = 59,
    HI_PIXEL_FORMAT_XYVU_PACKED_444 = 60,
    HI_PIXEL_FORMAT_YUV_SEMIPLANAR_411 = 61,
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_411 = 62,
    HI_PIXEL_FORMAT_YUV_PLANAR_411 = 63,
    HI_PIXEL_FORMAT_YVU_PLANAR_411 = 64,
    HI_PIXEL_FORMAT_YUV_PLANAR_440 = 65,
    HI_PIXEL_FORMAT_YVU_PLANAR_440 = 66,

    HI_PIXEL_FORMAT_RGB_888_PLANAR = 69,
    HI_PIXEL_FORMAT_BGR_888_PLANAR = 70,
    HI_PIXEL_FORMAT_HSV_888_PACKAGE = 71,
    HI_PIXEL_FORMAT_HSV_888_PLANAR = 72,
    HI_PIXEL_FORMAT_LAB_888_PACKAGE = 73,
    HI_PIXEL_FORMAT_LAB_888_PLANAR = 74,
    HI_PIXEL_FORMAT_S8C1 = 75,
    HI_PIXEL_FORMAT_S8C2_PACKAGE = 76,
    HI_PIXEL_FORMAT_S8C2_PLANAR = 77,
    HI_PIXEL_FORMAT_S16C1 = 78,
    HI_PIXEL_FORMAT_U8C1 = 79,
    HI_PIXEL_FORMAT_U16C1 = 80,
    HI_PIXEL_FORMAT_S32C1 = 81,
    HI_PIXEL_FORMAT_U32C1 = 82,
    HI_PIXEL_FORMAT_U64C1 = 83,
    HI_PIXEL_FORMAT_S64C1 = 84,

    HI_PIXEL_FORMAT_RGB_888_INT8 = 110,
    HI_PIXEL_FORMAT_BGR_888_INT8 = 111,
    HI_PIXEL_FORMAT_RGB_888_INT16 = 112,
    HI_PIXEL_FORMAT_BGR_888_INT16 = 113,
    HI_PIXEL_FORMAT_RGB_888_INT32 = 114,
    HI_PIXEL_FORMAT_BGR_888_INT32 = 115,
    HI_PIXEL_FORMAT_RGB_888_UINT16 = 116,
    HI_PIXEL_FORMAT_BGR_888_UINT16 = 117,
    HI_PIXEL_FORMAT_RGB_888_UINT32 = 118,
    HI_PIXEL_FORMAT_BGR_888_UINT32 = 119,
    HI_PIXEL_FORMAT_RGB_888_PLANAR_INT8  = 120,
    HI_PIXEL_FORMAT_BGR_888_PLANAR_INT8  = 121,
    HI_PIXEL_FORMAT_RGB_888_PLANAR_INT16 = 122,
    HI_PIXEL_FORMAT_BGR_888_PLANAR_INT16 = 123,
    HI_PIXEL_FORMAT_RGB_888_PLANAR_INT32 = 124,
    HI_PIXEL_FORMAT_BGR_888_PLANAR_INT32 = 125,
    HI_PIXEL_FORMAT_RGB_888_PLANAR_UINT16 = 126,
    HI_PIXEL_FORMAT_BGR_888_PLANAR_UINT16 = 127,
    HI_PIXEL_FORMAT_RGB_888_PLANAR_UINT32 = 128,
    HI_PIXEL_FORMAT_BGR_888_PLANAR_UINT32 = 129,
    HI_PIXEL_FORMAT_YUV400_UINT16 = 130,
    HI_PIXEL_FORMAT_YUV400_UINT32 = 131,
    HI_PIXEL_FORMAT_YUV400_UINT64 = 132,
    HI_PIXEL_FORMAT_YUV400_INT8   = 133,
    HI_PIXEL_FORMAT_YUV400_INT16  = 134,
    HI_PIXEL_FORMAT_YUV400_INT32  = 135,
    HI_PIXEL_FORMAT_YUV400_INT64  = 136,
    HI_PIXEL_FORMAT_YUV400_FP16 = 137,
    HI_PIXEL_FORMAT_YUV400_FP32 = 138,
    HI_PIXEL_FORMAT_YUV400_FP64 = 139,
    HI_PIXEL_FORMAT_YUV400_BF16 = 140,

    HI_PIXEL_FORMAT_YUV_SEMIPLANAR_440 = 1000,
    HI_PIXEL_FORMAT_YVU_SEMIPLANAR_440 = 1001,
    HI_PIXEL_FORMAT_FLOAT32 = 1002,
    HI_PIXEL_FORMAT_BUTT = 1003,

    HI_PIXEL_FORMAT_RGB_888_PLANAR_FP16 = 1004,
    HI_PIXEL_FORMAT_BGR_888_PLANAR_FP16 = 1005,
    HI_PIXEL_FORMAT_RGB_888_PLANAR_FP32 = 1006,
    HI_PIXEL_FORMAT_BGR_888_PLANAR_FP32 = 1007,
    HI_PIXEL_FORMAT_RGB_888_PLANAR_BF16 = 1008,
    HI_PIXEL_FORMAT_BGR_888_PLANAR_BF16 = 1009,
    HI_PIXEL_FORMAT_RGB_888_FP16 = 1010,
    HI_PIXEL_FORMAT_BGR_888_FP16 = 1011,
    HI_PIXEL_FORMAT_RGB_888_FP32 = 1012,
    HI_PIXEL_FORMAT_BGR_888_FP32 = 1013,
    HI_PIXEL_FORMAT_RGB_888_BF16 = 1014,
    HI_PIXEL_FORMAT_BGR_888_BF16 = 1015,

    HI_PIXEL_FORMAT_UNKNOWN = 10000
} hi_pixel_format;

typedef enum {
    HI_VIDEO_FORMAT_LINEAR = 0, // nature video line
    HI_VIDEO_FORMAT_TILE_64x16, // tile cell: 64pixel x 16line
    HI_VIDEO_FORMAT_BUTT
} hi_video_format;

typedef enum {
    HI_COMPRESS_MODE_NONE = 0, // no compress
    HI_COMPRESS_MODE_SEG, // compress unit is 256x1 bytes as a segment
    HI_COMPRESS_MODE_TILE, // compress unit is a tile
    HI_COMPRESS_MODE_HFBC,
    HI_COMPRESS_MODE_BUTT
} hi_compress_mode;

typedef enum {
    HI_COLOR_GAMUT_BT601 = 0,
    HI_COLOR_GAMUT_BT709,
    HI_COLOR_GAMUT_BT2020,
    HI_COLOR_GAMUT_USER,
    HI_COLOR_GAMUT_BUTT
} hi_color_gamut;

typedef enum {
    HI_DYNAMIC_RANGE_SDR8 = 0,
    HI_DYNAMIC_RANGE_SDR10,
    HI_DYNAMIC_RANGE_HDR10,
    HI_DYNAMIC_RANGE_HLG,
    HI_DYNAMIC_RANGE_SLF,
    HI_DYNAMIC_RANGE_XDR,
    HI_DYNAMIC_RANGE_BUTT
} hi_dynamic_range;

typedef enum {
    HI_VIDEO_FIELD_TOP = 0x1, // even field
    HI_VIDEO_FIELD_BOTTOM = 0x2, // odd field
    HI_VIDEO_FIELD_INTERLACED = 0x3, // two interlaced fields
    HI_VIDEO_FIELD_FRAME = 0x4, // frame

    HI_VIDEO_FIELD_BUTT
} hi_video_field;

typedef enum {
    HI_VIDEO_OUT_ORDER_DISPLAY = 0,
    HI_VIDEO_OUT_ORDER_DEC,
    HI_VIDEO_OUT_ORDER_BUTT
} hi_video_out_order;

typedef enum {
    HI_JPEG_RAW_FORMAT_YUV444 = 0,
    HI_JPEG_RAW_FORMAT_YUV422 = 1,
    HI_JPEG_RAW_FORMAT_YUV420 = 2,
    HI_JPEG_RAW_FORMAT_YUV440 = 3,
    HI_JPEG_RAW_FORMAT_YUV400 = 4,
    HI_JPEG_RAW_FORMAT_YUV411 = 5,
    HI_JPEG_RAW_FORMAT_MAX = 100
} hi_jpeg_raw_format;

typedef enum {
    HI_PNG_COLOR_FORMAT_GRAY  = 0x0,   // gray bitmap
    HI_PNG_COLOR_FORMAT_RGB   = 0x2,   // RGB bitmap
    HI_PNG_COLOR_FORMAT_CLUT  = 0x3,   // clut
    HI_PNG_COLOR_FORMAT_AGRAY = 0x4,   // gray bitmap with alpha
    HI_PNG_COLOR_FORMAT_ARGB  = 0x6,   // RGB bitmap with alpha
    HI_PNG_COLOR_FORMAT_BUTT  = 0x100
} hi_png_color_format;

typedef enum {
   HI_CSC_MATRIX_BT601_WIDE = 0,
   HI_CSC_MATRIX_BT601_NARROW,
   HI_CSC_MATRIX_BT709_WIDE,
   HI_CSC_MATRIX_BT709_NARROW,
   HI_CSC_MATRIX_BT2020_WIDE,
   HI_CSC_MATRIX_BT2020_NARROW,
   HI_CSC_MATRIX_USER = 100,
   HI_CSC_MATRIX_BUTT
} hi_csc_matrix;

typedef struct {
   hi_double csc_matrix_r0_c0;
   hi_double csc_matrix_r0_c1;
   hi_double csc_matrix_r0_c2;
   hi_double csc_matrix_r1_c0;
   hi_double csc_matrix_r1_c1;
   hi_double csc_matrix_r1_c2;
   hi_double csc_matrix_r2_c0;
   hi_double csc_matrix_r2_c1;
   hi_double csc_matrix_r2_c2;
   hi_double csc_bias_r0;
   hi_double csc_bias_r1;
   hi_double csc_bias_r2;
} hi_coefficient;

typedef struct {
    hi_coefficient yuv_to_rgb_coefficient;
    hi_coefficient rgb_to_yuv_coefficient;
} hi_csc_coefficient;

typedef struct {
    hi_u64 misc_info_phys_addr; // default allocated buffer
    hi_u64 jpeg_dcf_phys_addr;
    hi_u64 isp_info_phys_addr;
    hi_u64 low_delay_phys_addr;
    hi_u64 bnr_rnt_phys_addr;
    hi_u64 motion_data_phys_addr;
    hi_u64 frame_dng_phys_addr;

    hi_void* ATTRIBUTE misc_info_virt_addr; // misc info
    hi_void* ATTRIBUTE jpeg_dcf_virt_addr; // jpeg_dcf, used in JPEG DCF
    hi_void* ATTRIBUTE isp_info_virt_addr; // isp_frame_info, used in ISP debug, when get raw and send raw
    hi_void* ATTRIBUTE low_delay_virt_addr; // used in low delay
    hi_void* ATTRIBUTE bnr_mot_virt_addr; // used for 3dnr from bnr mot
    hi_void* ATTRIBUTE motion_data_virt_addr; // vpss 3dnr use: gme motion data, filter motion data, gyro data
    hi_void* ATTRIBUTE frame_dng_virt_addr;
} hi_video_supplement;

typedef struct {
    hi_u32              width;
    hi_u32              height;
    hi_video_field      field;
    hi_pixel_format     pixel_format;
    hi_video_format     video_format;
    hi_compress_mode    compress_mode;
    hi_dynamic_range    dynamic_range;
    hi_color_gamut      color_gamut;

    hi_u32              header_stride[HI_MAX_COLOR_COMPONENT];
    hi_u32              width_stride[HI_MAX_COLOR_COMPONENT];
    hi_u32              height_stride[HI_MAX_COLOR_COMPONENT];

    hi_u64              header_phys_addr[HI_MAX_COLOR_COMPONENT];
    hi_u64              phys_addr[HI_MAX_COLOR_COMPONENT];
    hi_void*            header_virt_addr[HI_MAX_COLOR_COMPONENT];
    hi_void*            virt_addr[HI_MAX_COLOR_COMPONENT];

    hi_u32              time_ref;
    hi_u64              pts;

    hi_u64              user_data[HI_MAX_USER_DATA_NUM];
    hi_u32              frame_flag; // frame_flag, can be OR operation.
    hi_video_supplement supplement;
} hi_video_frame;

typedef struct {
    hi_video_frame v_frame;
    hi_u32        pool_id;
    hi_mod_id      mod_id;
} hi_video_frame_info;

typedef struct {
    hi_u32 width;
    hi_u32 height;
} hi_video_size;

typedef struct {
    hi_u32 width;
    hi_u32 height;
    hi_u32 width_stride;
    hi_u32 height_stride;
    hi_u32 img_buf_size;
    union {
        hi_jpeg_raw_format pixel_format;
        hi_png_color_format png_pixel_format;
    };
    hi_u32 reserved[4];
} hi_img_info;

typedef struct {
    hi_u32            width;
    hi_u32            height;
    hi_u32            align;
    hi_data_bit_width bit_width;
    hi_pixel_format   pixel_format;
    hi_compress_mode  compress_mode;
} hi_pic_buf_attr;

typedef struct {
    hi_payload_type type;
    hi_u8 *ATTRIBUTE addr;   // stream address
    hi_u32 len;              // stream len
    hi_u64 pts;              // W; time stamp
    hi_s32 reserved[2];
} hi_img_stream;

typedef struct {
    hi_void* picture_address;
    hi_u32 picture_buffer_size;
    hi_u32 picture_width;
    hi_u32 picture_height;
    hi_u32 picture_width_stride;
    hi_u32 picture_height_stride;
    hi_pixel_format picture_format;
} hi_pic_info;

typedef struct {
    hi_s32 x;
    hi_s32 y;
    hi_u32 width;
    hi_u32 height;
} hi_rect;

/*
 * @brief alloc device memory for dvpp
 * @param [in] dev_id: the device id, set 0 in 1p device
 * @param [in] size: memory size
 * @param [out] dev_ptr: memory pointer
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_dvpp_malloc(hi_u32 dev_id, hi_void **dev_ptr, hi_u64 size);

/*
 * @brief free the memory requested through the hi_mpi_dvpp_malloc interface
 * @param [in] dev_ptr: memory pointer
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_dvpp_free(hi_void *dev_ptr);

/**
 * @brief query DVPP interface version
 * @param [out] major_version: DVPP interface major version
 * @param [out] minor_version: DVPP interface minor version
 * @param [out] patch_version: DVPP interface patch version
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_dvpp_get_version(hi_s32 *major_version, hi_s32 *minor_version, hi_s32 *patch_version);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif // #ifndef HI_DVPP_COMMON_H_
