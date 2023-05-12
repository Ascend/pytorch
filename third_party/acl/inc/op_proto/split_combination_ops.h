/*!
 * \file split_combination_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Packs the list of tensors in values into a tensor with rank one higher than each tensor in
* values, by packing them along the axis dimension. Given a list of length N of tensors of
* shape (A, B, C); if axis == 0 then the output tensor will have the shape (N, A, B, C) . \n

*@par Inputs:
* x: A list of N Tensors. Must be one of the following types: int8, int16, int32,
*     int64, uint8, uint16, uint32, uint64, float16, float32, bool . It's a dynamic input. \n

*@par Attributes:
*@li axis: A optional int, default value is 0.
*     Dimension along which to pack. The range is [-(R+1), R+1).
*@li N: A required int. Number of tensors . \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Pack.
*/
REG_OP(Pack)
    .DYNAMIC_INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(axis, Int, 0)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(Pack)




/**
*@brief Concatenates tensors along one dimension . \n

*@par Inputs:
* One input:
*x:Dynamic input. An NC1HWC0 or ND Tensor.
*Must be one of the following types: float16, float32, int32, int8, int16, int64, uint8, uint16, uint32, uint64

*@par Attributes:
*@li concat_dim: A required int8, int16, int32, or int64. Specifies the dimension along which to concatenate. No default value.
*@li N:  An optional int8, int16, int32, or int64. Specifies the number of elements in "x". No default value . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as "x" . \n

*@attention Constraints:
*@li "x" is a list of at least 2 "tensor" objects of the same type.
*@li "concat_dim" is in the range [-len(x.shape), len(x.shape)] . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Concat.
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use Concat instead.
*/
REG_OP(ConcatD)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
    .REQUIRED_ATTR(concat_dim, Int)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatD)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
