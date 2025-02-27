#pragma once
#include "third_party/acl/inc/aml/aml_fwk_detect.h"

namespace c10_npu {
namespace amlapi {
/**
 * This API is used to check whether AmlAicoreDetectOnline exist.
*/
bool IsExistAmlAicoreDetectOnline();

/**
 * This API is used to call AmlAicoreDetectOnline.
*/
AmlStatus AmlAicoreDetectOnlineFace(int32_t deviceId, const AmlAicoreDetectAttr *attr);

} // namespace amlapi
} // namespace c10_npu
