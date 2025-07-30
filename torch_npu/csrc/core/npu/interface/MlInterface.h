#pragma once
#include "third_party/acl/inc/aml/aml_fwk_detect.h"

namespace c10_npu {
namespace amlapi {
/**
 * This API is used to check whether AmlAicoreDetectOnline exist.
*/
bool IsExistAmlAicoreDetectOnline();

/**
 * This API is used to check whether AmlP2PDetectOnline exist.
*/
bool IsExistAmlP2PDetectOnline();

/**
 * This API is used to call AmlAicoreDetectOnline.
*/
AmlStatus AmlAicoreDetectOnlineFace(int32_t deviceId, const AmlAicoreDetectAttr *attr);

/**
 * This API is used to call AmlP2PDetectOnline.
*/
AmlStatus AmlP2PDetectOnlineFace(int32_t deviceId, void *comm, const AmlP2PDetectAttr *attr);

} // namespace amlapi
} // namespace c10_npu
