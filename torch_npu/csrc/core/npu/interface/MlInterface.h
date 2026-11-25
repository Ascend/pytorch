#pragma once

#include <aml/aml_fwk_detect.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace c10_npu {
namespace amlapi {
/**
 * This API is used to check whether AmlAicoreDetectOnline exist.
*/
C10_NPU_API bool IsExistAmlAicoreDetectOnline();

/**
 * This API is used to check whether AmlP2PDetectOnline exist.
*/
C10_NPU_API bool IsExistAmlP2PDetectOnline();

/**
 * This API is used to call AmlAicoreDetectOnline.
*/
C10_NPU_API AmlStatus AmlAicoreDetectOnlineFace(int32_t deviceId, const AmlAicoreDetectAttr *attr);

/**
 * This API is used to call AmlP2PDetectOnline.
*/
C10_NPU_API AmlStatus AmlP2PDetectOnlineFace(int32_t deviceId, void *comm, const AmlP2PDetectAttr *attr);

} // namespace amlapi
} // namespace c10_npu
