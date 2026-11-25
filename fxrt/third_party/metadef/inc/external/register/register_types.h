/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_REGISTER_REGISTER_TYPES_H_
#define INC_EXTERNAL_REGISTER_REGISTER_TYPES_H_

#if (defined(HOST_VISIBILITY)) && (defined(__GNUC__))
#define FMK_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define FMK_FUNC_HOST_VISIBILITY
#endif
#if (defined(DEV_VISIBILITY)) && (defined(__GNUC__))
#define FMK_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define FMK_FUNC_DEV_VISIBILITY
#endif
#ifdef __GNUC__

#ifdef NO_METADEF_ABI_COMPATIABLE
#define ATTRIBUTED_DEPRECATED(replacement)
#else
#define ATTRIBUTED_DEPRECATED(replacement) __attribute__((deprecated("Please use " #replacement " instead.")))
#endif

#else
#ifdef NO_METADEF_ABI_COMPATIABLE
#define ATTRIBUTED_DEPRECATED(replacement)
#else
#define ATTRIBUTED_DEPRECATED(replacement) __declspec(deprecated("Please use " #replacement " instead."))
#endif
#endif
namespace domi {

///
/// @ingroup domi
/// @brief original tensor type
///
typedef enum tagDomiTensorFormat {
  DOMI_TENSOR_NCHW = 0, // < NCHW
  DOMI_TENSOR_NHWC, // < NHWC
  DOMI_TENSOR_ND, // < Nd Tensor
  DOMI_TENSOR_NC1HWC0, // < NC1HWC0
  DOMI_TENSOR_FRACTAL_Z, // < FRACTAL_Z
  DOMI_TENSOR_NC1C0HWPAD,
  DOMI_TENSOR_NHWC1C0,
  DOMI_TENSOR_FSR_NCHW,
  DOMI_TENSOR_FRACTAL_DECONV,
  DOMI_TENSOR_BN_WEIGHT,
  DOMI_TENSOR_CHWN, // Android NN Depth CONV
  DOMI_TENSOR_FILTER_HWCK, // filter input tensor format
  DOMI_TENSOR_NDHWC,
  DOMI_TENSOR_NCDHW,
  DOMI_TENSOR_DHWCN, // 3D filter input tensor format
  DOMI_TENSOR_DHWNC,
  DOMI_TENSOR_RESERVED
} domiTensorFormat_t;
} // namespace domi

#endif // INC_EXTERNAL_REGISTER_REGISTER_TYPES_H_
