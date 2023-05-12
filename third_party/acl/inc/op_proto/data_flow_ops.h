/*!
 * \file data_flow_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_DATA_FLOW_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_DATA_FLOW_OPS_H_

#include <algorithm>
#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Enqueue a Tensor on the computation outfeed. \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: float16, float32,
float64, int8, int16, uint16, uint8, int32, int64, uint32, uint64,
bool, double, string. It's a dynamic input. \n

*@par Attributes:
*channel_name: name of operator channel, default "". \n

*@attention Constraints:
*The implementation for OutfeedEnqueueOp on Ascend uses AICPU, with bad performance.

*@par Third-party framework compatibility
*@li compatible with tensorflow OutfeedEnqueueOp operator.
*/
REG_OP(OutfeedEnqueueOp)
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8,
      DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
      DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
  .ATTR(channel_name, String, "")
  .OP_END_FACTORY_REG(OutfeedEnqueueOp)

REG_OP(OutfeedEnqueueOpV2)
  .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8,
      DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
      DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
  .INPUT(tensor_name, TensorType({DT_STRING}))
  .ATTR(channel_name, String, "")
  .OP_END_FACTORY_REG(OutfeedEnqueueOpV2)

}   // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_DATA_FLOW_OPS_H_

