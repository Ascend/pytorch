#ifndef __INCLUDE_CUSTOM_OP_API_H__
#define __INCLUDE_CUSTOM_OP_API_H__

#include "ops/cpu/cpu_custom_operator.h"
#include "ops/custom_op_register.h"

#ifdef ASCEND_CUSTOM_OP
#include "ops/ascend/aclnn/aclnn_custom_operator.h"
#include "ops/ascend/aclnn/aclnn_custom_view_operator.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#endif

#endif // __INCLUDE_CUSTOM_OP_API_H__
