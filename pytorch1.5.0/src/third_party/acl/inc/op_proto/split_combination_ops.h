/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file split_combination_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Splits a tensor along dimension "split_dim" into "num_split" smaller tensors . \n

*@par Inputs:
* Two inputs, including:
*@li x: An ND Tensor.
*Must be one of the types:float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8, int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32.
*@li split_dim: Must be the following type:int32. Specifies the dimension along which to split . \n

*@par Attributes:
*num_split: A required int32. Specifies the number of output tensors. No default value . \n

*@par Outputs:
*y: Dynamic output.A list of output tensors. Has the same type and format as "x" . \n

*@attention Constraints:
*@li "num_split" is greater than or equals to 1.
*@li "num_split" is divisible by the size of dimension "split_dim".
*@li "split_dim" is in the range [-len(x.shape), (x.shape)-1] . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Split.
*/
REG_OP(Split)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(Split)

/**
*@brief Splits a tensor along dimension "split_dim" into "num_split" smaller tensors . \n

*@par Inputs:
* One input:
*: An ND Tensor.
*Must be one of the following types: float16, float32, int32, int8, int16, int64, uint8, uint16, uint32, uint64

*@par Attributes:
*@li split_dim: A required int32. Specifies the dimension along which to split. No default value.
*@li num_split: A required int32. Specifies the number of output tensors. No default value . \n

*@par Outputs:
*y:Dynamic output. A list of output tensors. Has the same type and format as "x" . \n

*@attention Constraints:
*@li "num_split" is greater than or equals to 1.
*@li "num_split" is divisible by the size of dimension "split_dim".
*@li "split_dim" is in the range [-len(x.shape), (x.shape)-1] . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Split.

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Split instead.
*/
REG_OP(SplitD)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                    DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                             DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(split_dim, Int)
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitD)

/**
*@brief Splits a tensor along dimension "split_dim" into "num_split" smaller tensors according to "size_splits" . \n

*@par Inputs:
* Three inputs, including:
*@li x: An ND Tensor.
*Must be one of the types:float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8, int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32.
*@li size_splits: Must be one of the types:int32, int64. Specifies a list containing the sizes of each output tensor along the split dimension.
*@li split_dim: Must be the following type:int32. Specifies the dimension along which to split . \n

*@par Attributes:
*num_split: A required int32. Specifies the number of output tensors. No default value . \n

*@par Outputs:
*y:  Dynamic output.A list of output tensors. Has the same type and format as "x" . \n

*@attention Constraints:
*@li Each element in "size_splits" is greater than or equal to 1.
*@li "size_splits" and "num_split" have the same length.
*@li The elements in "size_splits" sum to the size of dimension "split_dim" . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SplitV.
*/
REG_OP(SplitV)
    .INPUT(x, TensorType::BasicType())
    .INPUT(size_splits, TensorType::IndexNumberType())
    .INPUT(split_dim, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitV)

/**
*@brief Splits a tensor along dimension "split_dim" into "num_split" smaller tensors according to "size_splits" . \n

*@par Inputs:
* One input:
* x: An ND Tensor.
*Must be one of the following types: float16, float32, int32, int8, int16, int64, uint8, uint16, uint32, uint64

*@par Attributes:
*@li size_splits: A required list of int32. Specifies a list containing the sizes of each output tensor along the split dimension.
*@li split_dim: A required int32. Specifies the dimension along which to split. No default value.
*@li num_split: A required int32. Specifies the number of output tensors. No default value . \n

*@par Outputs:
*y: Dynamic output.A list of output tensors. Has the same type and format as "x" . \n

*@attention Constraints:
*@li Each element in "size_splits" is greater than or equal to 1.
*@li "size_splits" and "num_split" have the same length.
Under the caffe framework, the conversion of slice_point through the cut point to cut segment is mapped to size_splits.
*@li The elements in "size_splits" sum to the size of dimension "split_dim".
Under the caffe framework,size_splits or axis transformat to split_dim.Only one can effect.
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator SplitV.

* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use SplitV instead.
*/
REG_OP(SplitVD)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                    DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                             DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(size_splits, ListInt)
    .REQUIRED_ATTR(split_dim, Int)
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitVD)

/**
*@brief Concatenates a list of N tensors along the first dimension.
*@par Inputs:
* One input, including:
* values: A list of Tensors. Must be one of the following types: int8, int16, int32,
*     int64, uint8, uint16, uint32, uint64, float16, float32.
*     Tensors to be concatenated. All must have size 1 in the first dimension and same shape.
*     It's a dynamic input. \n

*@par Attributes:
* @li shape: A required list of ints.
* @li N: The numble of dynamic_input "values" . \n

*@par Outputs:
*output_data: The concatenated tensor with same type as "values".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ParallelConcat.
*/
REG_OP(ParallelConcat)
    .DYNAMIC_INPUT(values, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
    .OUTPUT(output_data, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ParallelConcat)

/**
*@brief Concatenates tensors along one dimension . \n

*@par Inputs:
* One input:
*x: Dynamic input.An NC1HWC0 or ND Tensor.
*Must be one of the following types: float16, float32, int32, int8, int16, int64, uint8, uint16, uint32, uint64

*@par Attributes:
*concat_dim: A required int8, int16, int32, or int64. Specifies the dimension along which to concatenate. No default value.
*N: An attribute int8, int16, int32, or int64. Specifies the number of elements in "x". Defaults to "1".

*@par Outputs:
*y: A Tensor. Has the same type and format as "x" . \n

*@attention Constraints:
*@li "x" is a list of at least 2 "tensor" objects of the same type.
*@li "concat_dim" is in the range [-len(x.shape), len(x.shape)] . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ConcatV2.
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use ConcatV2 instead.
*/
REG_OP(ConcatV2D)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_INT64, DT_UINT64, DT_UINT32, DT_INT16, DT_UINT16, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_INT64, DT_UINT64, DT_UINT32, DT_INT16, DT_UINT16, DT_UINT8}))
    .REQUIRED_ATTR(concat_dim, Int)
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatV2D)

/**
*@brief Concatenates tensors along one dimension . \n

*@par Inputs:
* Two inputs, including:
*@li Dynamic input "x" is An NC1HWC0 or ND Tensor.
*Must be one of the following types: float16, float32, int32, int8, int16, int64, uint8, uint16, uint32, uint64
*@li concat_dim: An int32, or int64. Specifies the dimension along which to concatenate . \n

*@par Attributes:
*N: An optional int8, int16, int32, or int64. Specifies the number of elements in "x". No default value . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as "x" . \n

*@attention Constraints:
* "x" is a list of at least 2 "tensor" objects of the same type . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator ConcatV2.
*/
REG_OP(ConcatV2)
    .DYNAMIC_INPUT(x, TensorType::BasicType())
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatV2)

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

/**
*@brief Concatenates tensors along one dimension . \n

*@par Inputs:
* Two inputs, including:
*@li x: Dynamic input.An NC1HWC0 or ND Tensor.
*Must be one of the following types: float16, float32, double, int32,
*     uint8, int16, int8, complex64, int64, qint8, quint8, qint32, uint16,
*     complex128, uint32, uint64, qint16, quint16.
*@li concat_dim: An int32, or int64. Specifies the dimension along which to concatenate . \n

*@par Attributes:
*N: An optional int8, int16, int32, or int64. Specifies the number of elements in "x" . \n

*@par Outputs:
*y: A Tensor. Has the same type and format as "x" . \n

*@attention Constraints:
*@li "x" is a list of at least 2 "tensor" objects of the same type.
*@li "concat_dim" is in the range [-len(x.shape), len(x.shape)] . \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Concat.
*/
REG_OP(Concat)
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .DYNAMIC_INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(Concat)

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
*@brief Computes offsets of concat inputs within its output . \n

*@par Inputs:
*Two inputs, including:
* @li concat_dim: A Tensor of type int32.
* @li x: A list of 1D Tensor objects of type int32 . It's a dynamic input. \n

*@par Attributes:
*N: A required int . \n

*@par Outputs:
*y: A Tensor list with same type as "x" . It's a dynamic output. \n

*@par Third-party framework compatibility
*@ Compatible with the TensorFlow operator ConcatOffset.
*/
REG_OP(ConcatOffset)
    .INPUT(concat_dim, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(x, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ConcatOffset)

/**
*@brief Computes offsets of concat inputs within its output . \n

*@par Inputs:
*Two inputs, including:
* @li concat_dim: A Tensor of type int32.
* @li x: A list of 1D Tensor objects of type int32 . It's a dynamic input. \n

*@par Attributes:
*@li Concat_dim: A required int. Must be within the rank of input "x".
*@li N: A required int . \n

*@par Outputs:
*y: A Tensor list with same type as "x" . It's a dynamic output. \n

*@par Third-party framework compatibility
*@ Compatible with the TensorFlow operator ConcatOffset.
*@par Restrictions:
*Warning: THIS FUNCTION IS DEPRECATED. Please use ConcatOffset instead.
*/
REG_OP(ConcatOffsetD)
    .DYNAMIC_INPUT(x, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(concat_dim, Int)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(ConcatOffsetD)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SPLIT_COMBINATION_OPS_H_
