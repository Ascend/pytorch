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

#ifndef HI_VPC_H_
#define HI_VPC_H_

#include "hi_dvpp_common.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif // #ifdef __cplusplus

#define VPC_MAX_CHN_NUM 512

/*********************************************************************************************/
/* invlalid channel ID */
#define HI_ERR_VPC_INVALID_CHN_ID 0xA0078002
/* at lease one parameter is illagal ,eg, an illegal enumeration value  */
#define HI_ERR_VPC_ILLEGAL_PARAM  0xA0078003
/* channel exists */
#define HI_ERR_VPC_EXIST          0xA0078004
/* the channel is not existed  */
#define HI_ERR_VPC_UNEXIST        0xA0078005
/* using a NULL point */
#define HI_ERR_VPC_NULL_PTR       0xA0078006
/* try to enable or initialize system,device or channel, before configuring attribute */
#define HI_ERR_VPC_NOT_CFG        0xA0078007
/* operation is not supported by NOW */
#define HI_ERR_VPC_NOT_SUPPORT    0xA0078008
/* operation is not permitted ,eg, try to change stati attribute */
#define HI_ERR_VPC_NOT_PERM       0xA0078009
/* failure caused by malloc memory */
#define HI_ERR_VPC_NO_MEM         0xA007800C
/* failure caused by malloc buffer */
#define HI_ERR_VPC_NO_BUF         0xA007800D
/* no data in buffer */
#define HI_ERR_VPC_BUF_EMPTY      0xA007800E
/* no buffer for new data */
#define HI_ERR_VPC_BUF_FULL       0xA007800F
/* system is not ready,had not initialized or loaded */
#define HI_ERR_VPC_SYS_NOT_READY  0xA0078010
/* bad address,  eg. used for copy_from_user & copy_to_user   */
#define HI_ERR_VPC_BAD_ADDR       0xA0078011
/* system busy */
#define HI_ERR_VPC_BUSY           0xA0078012
/* system err */
#define HI_ERR_VPC_SYS_ERROR      0xA0078015

typedef hi_s32 hi_vpc_chn;

typedef enum {
    HI_BORDER_CONSTANT = 0,
    HI_BORDER_REPLICATE,
    HI_BORDER_REFLECT,
    HI_BORDER_REFLECT_101,
    HI_BORDER_WRAP
} hi_vpc_bord_type;

typedef struct {
    hi_double val[4];
} hi_vpc_scalar;

typedef struct {
    hi_u32 top;
    hi_u32 bottom;
    hi_u32 left;
    hi_u32 right;
    hi_vpc_bord_type border_type;
    hi_vpc_scalar scalar_value;
} hi_vpc_make_border_info;

typedef struct {
    hi_void* picture_address;
    hi_u32 picture_buffer_size;
    hi_u32 picture_width;
    hi_u32 picture_height;
    hi_u32 picture_width_stride;
    hi_u32 picture_height_stride;
    hi_pixel_format picture_format;
} hi_vpc_pic_info;

typedef struct {
    hi_u32 top_offset;
    hi_u32 left_offset;
    hi_u32 crop_width;
    hi_u32 crop_height;
} hi_vpc_crop_region;

typedef struct {
    hi_vpc_pic_info dest_pic_info;
    hi_vpc_crop_region crop_region;
} hi_vpc_crop_region_info;

typedef struct {
    hi_u32 resize_width;
    hi_u32 resize_height;
    hi_u32 interpolation;
} hi_vpc_resize_info;

typedef struct {
    hi_vpc_pic_info dest_pic_info;
    hi_vpc_crop_region crop_region;
    hi_vpc_resize_info resize_info;
} hi_vpc_crop_resize_region;

typedef struct {
    hi_vpc_pic_info dest_pic_info;
    hi_vpc_crop_region crop_region;
    hi_vpc_resize_info resize_info;
    hi_u32 dest_top_offset;
    hi_u32 dest_left_offset;
} hi_vpc_crop_resize_paste_region;

typedef struct {
    hi_vpc_pic_info dest_pic_info;
    hi_vpc_crop_region crop_region;
    hi_vpc_resize_info resize_info;
    hi_u32 dest_top_offset;
    hi_u32 dest_left_offset;
    hi_vpc_bord_type border_type;
    hi_vpc_scalar scalar_value;
} hi_vpc_crop_resize_border_region;

typedef struct {
    hi_vpc_pic_info dest_pic_info;
    hi_vpc_crop_region crop_region;
    hi_vpc_resize_info resize_info1; // First bilinear core resize info
    hi_vpc_resize_info resize_info2; // Second bilinear core resize info
    hi_u32 dest_top_offset;
    hi_u32 dest_left_offset;
    hi_u32 reserved[2]; // RW; reserved init 0
} hi_vpc_crop_resize_resize_paste_region;

typedef struct {
    hi_s32 attr; // RW; reserved
    hi_u32 pic_width; // RW; max pic width
    hi_u32 pic_height; // RW; max pic height
} hi_vpc_chn_attr;

typedef struct {
    hi_u32 histogram_y_or_r[256]; // 256 level statistics of Y or R component
    hi_u32 histogram_u_or_g[256]; // 256 level statistics of U or G component
    hi_u32 histogram_v_or_b[256]; // 256 level statistics of V or B component
} hi_vpc_histogram_config;

typedef struct {
    hi_u8 map_value_y_or_r[256]; // remap value of Y or R component
    hi_u8 map_value_u_or_g[256]; // remap value of U or G component
    hi_u8 map_value_v_or_b[256]; // remap value of V or B component
} hi_vpc_lut_remap;

typedef struct {
    hi_u32 alpha; // Alpha indicates how opaque each pixel is
    hi_u32 attr[4]; // RW; reserved
} hi_csc_conf;

typedef struct {
    hi_u32 kernel_size;
} hi_median_blur_config;

typedef struct {
    hi_vpc_pic_info src;
    hi_vpc_pic_info dst;
    hi_median_blur_config median_blur_cfg;
} hi_median_blur_param;

typedef struct {
    hi_s32 x;
    hi_s32 y;
} hi_point;

typedef enum {
    MORPH_RECT = 0,     // 矩形
    MORPH_CROSS = 1,    // 交叉形 暂不支持
    MORPH_ELLIPSE = 2,  // 椭圆形 暂不支持
    MORPH_MAX = 100     // 暂不支持
} hi_morph_shapes;

typedef struct {
    hi_u32 width;
    hi_u32 height;
} hi_size;

typedef struct {
    hi_size kernel_size;
    hi_morph_shapes morph_shapes;
    hi_point anchor;                // reserved, 需要设置成（-1，-1）
    hi_u32 iterations;              // 迭代次数 blur只支持设置成1，erode/dilate 最大支持100
    hi_vpc_bord_type border_type;
    hi_vpc_scalar scalar_value;
} hi_blur_config;

typedef struct {
    hi_vpc_pic_info src;
    hi_vpc_pic_info dst;
    hi_blur_config blur_cfg;
} hi_blur_param;

typedef struct {
    hi_size kernel_size;
    hi_double sigma_x;
    hi_double sigma_y;
    hi_vpc_bord_type border_type;
    hi_vpc_scalar scalar_value;
    hi_u32 reserved[2];
} hi_gaussian_blur_config;

typedef struct {
    hi_vpc_pic_info src;
    hi_vpc_pic_info dst;
    hi_gaussian_blur_config gaussian_blur_cfg;
} hi_gaussian_blur_param;

typedef struct {
    hi_double filter[5][5]; // convolution kernel
    hi_size kernel_size;    // kernel size
    hi_point anchor;        // reserved, 需要设置成（-1，-1）
    hi_double delta;        // optional value `delta` added to the filtered pixels before storing them in dst
    hi_vpc_bord_type border_type;
    hi_vpc_scalar scalar_value;
    hi_u32 reserved[2];
} hi_filter_2d_config;

typedef struct {
    hi_vpc_pic_info src;
    hi_vpc_pic_info dst;
    hi_filter_2d_config filter_2d_cfg;
} hi_filter_2d_param;

typedef struct {
    hi_u32 pic1_index;
    hi_u32 pic2_index;
    hi_float weight1;
    hi_float weight2;
    hi_float offset1_x;
    hi_float offset1_y;
    hi_float offset2_x;
    hi_float offset2_y;
} hi_stiching_ipm_table;

typedef struct {
    hi_u32 src_pic_width;
    hi_u32 src_pic_height;
    hi_u32 dest_pic_width;
    hi_u32 dest_pic_height;
    hi_u32 ipm_table_len;
    hi_stiching_ipm_table* ipm_table_address;
} hi_stiching_ipm_param;

typedef enum {
    OVERLAP_HISTOGRAM = 0,
    GLOBAL_HISTOGRAM,
} hi_stitching_histogram_type;

typedef enum  {
    GAIN_NONE = 0,
    GAIN_LUT,
} hi_stitching_gain_type;

typedef struct {
    hi_stitching_histogram_type hist_type;
    hi_stitching_gain_type gain_type;
    hi_void *user_data;
    hi_s32 (*calulate_gain_callback)(
        hi_void *user_data,
        hi_void *hist_config,
        hi_u32 hist_count,
        hi_void *gain,
        hi_u32 gain_size);
} hi_stitching_gain_param;

typedef struct {
    hi_stiching_ipm_param imp_conf;
    hi_stitching_gain_param gain_conf;
    hi_u32 reserved[8];
} hi_roundview_stitching_param;

typedef struct {
    hi_vpc_pic_info *dest_pic;
    hi_u32 count;
    hi_vpc_pic_info *source_pic[];
} hi_roundview_stitching_pic_param;

typedef struct {
    hi_u32 pic_index;
    hi_u32 overlap_pic_index;
    hi_vpc_histogram_config histogram_info;
} hi_stitching_histogram_param;

typedef enum {
    HI_ROTATION_90 = 0,
    HI_ROTATION_180 = 1,
    HI_ROTATION_270 = 2,
} hi_rotation;

typedef struct {
    hi_vpc_pic_info src;
    hi_vpc_pic_info dst;
    hi_rotation angle;
} hi_rotate_param;

typedef enum {
    HI_BLK_SIZE_4 = 0,    /* block size 4*4 */
    HI_BLK_SIZE_8,        /* block size 8*8 */
    HI_BLK_SIZE_16,       /* block size 16*16 */
    HI_BLK_SIZE_32,       /* block size 32*32 */
    HI_BLK_SIZE_64,       /* block size 64*64 */
    HI_BLK_SIZE_128,      /* block size 128*128 */
} hi_blk_size;

typedef struct {
    hi_blk_size blk_size;
    hi_rect rect;
} hi_mosaic;

typedef struct {
    hi_vpc_pic_info src;
    hi_vpc_pic_info dst;
    hi_u32 count;
    hi_mosaic *mosaic;
} hi_mosaic_param;

typedef enum {
    HI_COVER_RECT = 0,
    HI_COVER_QUAD,
} hi_cover_type;

typedef struct {
    hi_bool is_solid;
    hi_u32 thick;
    hi_point point[4]; // 四边形
} hi_quad_cover;

typedef struct {
    hi_cover_type type;
    union {
        hi_rect rect;
        hi_quad_cover quad;
    };
    hi_u32 color;
} hi_cover;

typedef struct {
    hi_vpc_pic_info src;
    hi_vpc_pic_info dst;
    hi_u32 count;
    hi_cover *cover;
} hi_cover_param;

typedef struct {
    hi_point start_point;
    hi_point end_point;
    hi_u32 thick;
    hi_u32 color;
} hi_line;

typedef struct {
    hi_vpc_pic_info src;
    hi_vpc_pic_info dst;
    hi_u32 count;
    hi_line *line;
} hi_line_param;

typedef enum {
    HI_OSD_INVERTED_COLOR_NONE = 0,     /* Not invert. */
    HI_OSD_INVERTED_COLOR_RGB,          /* Invert rgb. */
    HI_OSD_INVERTED_COLOR_ALPHA,        /* Invert alpha. */
    HI_OSD_INVERTED_COLOR_ALL,          /* Invert rgb and alpha. */
} hi_osd_inverted_color;

typedef struct {
    hi_rect rect;
    hi_pixel_format pixel_format;
    hi_void* picture_address;
    hi_u32 stride;
    hi_u32 bg_alpha;
    hi_u32 fg_alpha;
    hi_osd_inverted_color osd_inverted_color;
} hi_osd;

typedef struct {
    hi_vpc_pic_info src;
    hi_vpc_pic_info dst;
    hi_u32 count;
    hi_osd *osd;
    hi_u32 clut[16];
} hi_osd_param;

/*
 * @brief median filter
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] median_blur_param: pointer of median blur param
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_median_blur(hi_vpc_chn chn, const hi_median_blur_param* median_blur_param, hi_u32* task_id,
    hi_s32 milli_sec);

/*
 * @brief erode filter
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] erode_param: pointer of erode blur param
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_erode(hi_vpc_chn chn, const hi_blur_param* erode_param, hi_u32* task_id, hi_s32 milli_sec);

/*
 * @brief dilate filter
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] dilate_param: pointer of dilate blur param
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_dilate(hi_vpc_chn chn, const hi_blur_param* dilate_param, hi_u32* task_id, hi_s32 milli_sec);

/*
 * @brief blur filter
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] blur_param: pointer of blur param
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_blur(hi_vpc_chn chn, const hi_blur_param* blur_param, hi_u32* task_id, hi_s32 milli_sec);

/*
 * @brief gaussian filter
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] gaussian_blur_param: pointer of gaussian blur param
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_gaussian_blur(hi_vpc_chn chn, const hi_gaussian_blur_param* gaussian_blur_param, hi_u32* task_id,
    hi_s32 milli_sec);

/*
 * @brief median filter
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] filter_2d_param: pointer of filter 2D param
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_filter2d(hi_vpc_chn chn, const hi_filter_2d_param* filter_2d_param, hi_u32* task_id,
    hi_s32 milli_sec);

/*
 * @brief imgae rotate
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] rotate_param: pointer of rotate param
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_rotate(hi_vpc_chn chn, const hi_rotate_param* rotate_param, hi_u32* task_id, hi_s32 milli_sec);

/*
 * @brief draw mosaic
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] mosaic_param: pointer of mosaic param
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_draw_mosaic(hi_vpc_chn chn, const hi_mosaic_param* mosaic_param, hi_u32* task_id, hi_s32 milli_sec);

/*
 * @brief draw cover
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] cover_param: pointer of cover param
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_draw_cover(hi_vpc_chn chn, const hi_cover_param* cover_param, hi_u32* task_id, hi_s32 milli_sec);

/*
 * @brief draw line
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] line_param: pointer of line param
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_draw_line(hi_vpc_chn chn, const hi_line_param* line_param, hi_u32* task_id, hi_s32 milli_sec);

/*
 * @brief draw osd
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] osd_param: pointer of osd param
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_draw_osd(hi_vpc_chn chn, hi_osd_param* osd_param, hi_u32* task_id, hi_s32 milli_sec);

/*
 * @brief create vpc channel
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] attr: pointer of vpc channel attribute
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_create_chn(hi_vpc_chn chn, const hi_vpc_chn_attr *attr);

/*
 * @brief create system vpc channel for single channel multi-core acceleration
 * @param [out] chnl: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] attr: pointer of vpc channel attribute
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_sys_create_chn(hi_vpc_chn *chnl, const hi_vpc_chn_attr *attr);

/*
 * @brief destroy vpc channel
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_destroy_chn(hi_vpc_chn chn);

/*
 * @brief crop
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] crop_info: array of vpc crop region info
 * @param [in] count: array length of crop_info
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_crop(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic, hi_vpc_crop_region_info crop_info[],
    hi_u32 count, hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief resize
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] dest_pic: pointer of vpc dest picture info
 * @param [in] fx: width resize info
 * @param [in] fy: height resize info
 * @param [in] interpolation: resize mode, support bilinear/nearest neighbor
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_resize(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic, hi_vpc_pic_info *dest_pic,
    hi_double fx, hi_double fy, hi_u32 interpolation, hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief crop and resize
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] crop_resize_info: array of vpc crop_resize region info
 * @param [in] count: array length of crop_resize_info
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_crop_resize(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic,
    hi_vpc_crop_resize_region crop_resize_info[], hi_u32 count, hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief crop and resize and paste
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] crop_resize_paste_info: array of vpc crop_resize_paste region info
 * @param [in] count: array length of crop_resize_paste_info
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_crop_resize_paste(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic,
    hi_vpc_crop_resize_paste_region crop_resize_paste_info[], hi_u32 count, hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief crop and resize and paste interface for batch pictures batch regions
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: array of pointer of vpc source picture info
 * @param [in] crop_resize_paste_info: array of vpc crop_resize_paste region info
 * @param [in] count: the num of regions correspond to source_pic
 * @param [in] pic_num: number os pictures
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_batch_crop_resize_paste(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic[], hi_u32 pic_num,
    hi_vpc_crop_resize_paste_region crop_resize_paste_info[], hi_u32 count[], hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief crop and resize and make border
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] crop_resize_make_border_info: array of vpc crop_resize_make_border region info
 * @param [in] count: array length of crop_resize_make_border
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_crop_resize_make_border(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic,
    hi_vpc_crop_resize_border_region crop_resize_make_border_info[], hi_u32 count, hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief crop and resize and make border for batch pictures and batch regions
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: arrary of pointer of vpc source picture info
 * @param [in] crop_resize_make_border_info: array of vpc crop_resize_make_border region info
 * @param [in] count: the num of regions correspond to source_pic
 * @param [in] pic_num: number os pictures
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_batch_crop_resize_make_border(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic[], hi_u32 pic_num,
    hi_vpc_crop_resize_border_region crop_resize_make_border_info[], hi_u32 count[], hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief convert color
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] dest_pic: pointer of vpc dest picture info
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_convert_color(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic, hi_vpc_pic_info *dest_pic,
    hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief crop and resize and resize and paste
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] crop_resize_resize_paste_info: array of vpc crop_resize_resize_paste region info
 * @param [in] count: array length of crop_resize_resize_paste_info
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: task id of crop_resize_resize_paste interface
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_crop_resize_resize_paste(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic,
    hi_vpc_crop_resize_resize_paste_region crop_resize_resize_paste_info[],
    hi_u32 count, hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief convert color to yuv420sp
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] dest_pic: pointer of vpc dest picture info
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_convert_color_to_yuv420(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic, hi_vpc_pic_info *dest_pic,
    hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief convert color with specified alpha channel
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] dest_pic: pointer of vpc dest picture info
 * @param [in] conf: pointer of csc config, support user config alpha channel
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_convert_color_v2(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic, hi_vpc_pic_info *dest_pic,
    hi_csc_conf *conf, hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief border padding
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] dest_pic: pointer of vpc dest picture info
 * @param [in] make_border_info: boundary fill information
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_copy_make_border(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic, hi_vpc_pic_info *dest_pic,
    hi_vpc_make_border_info make_border_info, hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief multi-level pyramid down
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] dest_pic[]: pointer array of vpc dest picture info
 * @param [in] filterLevel: pyramid level [1, 4]
 * @param [in] gaussian_filter: convolution kernel of 5x5
 * @param [in] make_border_info: boundary padding algorithm and boundary pixel value info
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_pyrdown(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic,
    hi_vpc_pic_info dest_pic[], hi_u32 filter_level, hi_s8 gaussian_filter[][5],
    hi_u16 divisor, hi_vpc_make_border_info make_border_info, hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief calculate histogram
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] hist_config: histogram result
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_calc_hist(hi_vpc_chn chn, const hi_vpc_pic_info *source_pic, hi_vpc_histogram_config *hist_config,
    hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief equalization image
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] source_pic: pointer of vpc source picture info
 * @param [in] dest_pic: pointer of vpc dest picture info
 * @param [in] lut_remap: remap value of YUV or RGB component
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @param [out] task_id: pointer of the task ID, used to distinguish tasks.
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_equalize_hist(hi_vpc_chn chn, const hi_vpc_pic_info* source_pic, hi_vpc_pic_info *dest_pic,
    const hi_vpc_lut_remap *lut_remap, hi_u32 *task_id, hi_s32 milli_sec);

/*
 * @brief query whether the task has been completed, base on task_id
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] task_id: task id generated by calling image processing interface
 * @param [in] milli_sec: -1 is block, 0 is no block, other positive number is timeout
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_get_process_result(hi_vpc_chn chn, hi_u32 task_id, hi_s32 milli_sec);

/*
 * @brief : set roundview stitching parameters
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] stitch_param: pointer of the HI_ROUNDVIEW_STITCH_PARA structure
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_set_roundview_stitching_param(hi_vpc_chn chn, const hi_roundview_stitching_param *stitch_param);

/*
 * @brief : get roundview stitching parameters
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [out] stitch_param: pointer of the HI_ROUNDVIEW_STITCH_PARA structure that you want to get
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_get_roundview_stitching_param(hi_vpc_chn chn, hi_roundview_stitching_param *stitch_param);

/*
 * @brief : deliver a roundview stitching task
 * @param [in] chn: vpc channel id [0, VPC_MAX_CHN_NUM)
 * @param [in] roundview_stitching_pic_param: pointer of structure that contains input and output pictures
 * @return success: return 0
 *         fail: return error number
 */
hi_s32 hi_mpi_vpc_roundview_stitching(hi_vpc_chn chn, hi_roundview_stitching_pic_param *roundview_stitching_pic_param,
    hi_u32 *task_id, hi_s32 milli_sec);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif // #ifdef __cplusplus

#endif // #ifndef HI_VPC_H_
