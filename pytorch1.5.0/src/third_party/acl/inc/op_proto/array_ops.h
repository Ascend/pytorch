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
 * \file array_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ARRAY_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ARRAY_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Applies lower_bound(sorted_search_values, values) along each row. \n

*@par Inputs:
*The input sorted_x and values can be one-dimensional vector. Inputs include:
* @li sorted_x:A `Tensor`. 2-D Tensor where each row is ordered.
* @li values:A `Tensor`. Must have the same type as `sorted_x`. \n

*@par Attributes:
*out_type:An optional `DType` from: `int32, int64`.
Defaults to `int32`. \n

*@par Outputs:
*y: A `Tensor` of type `out_type`. \n

*@attention Constraints:
*The implementation for LowerBound on Ascend uses AI CPU, with bad performance. \n

*@par Quantization supported or not
*Not supported
*@par Quantized inference supported or not
*Supported
*@par L2 convergence supported or not
*@par Multiple batches supported or not \n

*@par Third-party framework compatibility
*Compatible with tensorflow Operator LowerBound.
*/

REG_OP(LowerBound)
    .INPUT(sorted_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, \
        DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_type, Type, DT_INT32)
    .OP_END_FACTORY_REG(LowerBound)

/**
*@brief Reverses variable length slices. \n

*@par Inputs:
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper"
are 0D scalars.
* @li x: A Tensor. The input to reverse.
* @li seq_lengths: A 1D Tensor of type int32 or int64. \n

*@par Attributes:
*@li seq_dim: An optional int. The dimension along which
reversal is performed.
*@li batch_dim: An optional int. Defaults to "0". The dimension along which
reversal is performed. \n

*@par Outputs:
*y: A rank k tensor. Has the same shape as input. The extracted banded tensor. \n

*@attention Constraints:
*ReverseSequence runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ReverseSequence.
*/

REG_OP(ReverseSequence)
    .INPUT(x,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(seq_lengths, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y,
        TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_BOOL, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
    .REQUIRED_ATTR(seq_dim, Int)
    .ATTR(batch_dim, Int, 0)
    .OP_END_FACTORY_REG(ReverseSequence)

/**
*@brief Copies a tensor setting everything outside a central band in each innermost matrix. \n

*@par Inputs:
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper"
are 0D scalars.
* @li x: A rank k tensor.
* @li num_lower: A 0D tensor. Number of superdiagonals to keep. If negative,
keeps entire upper triangle.
* @li num_upper: A 0D tensor. Number of superdiagonals to keep. If negative,
keeps entire upper triangle. \n

*@par Outputs:
*y: A rank k tensor. Has the same shape as input. The extracted banded tensor. \n

*@attention Constraints:
*MatrixBandPart runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MatrixBandPart.
*/

REG_OP(MatrixBandPart)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, \
           DT_INT16, DT_UINT16, DT_INT32, DT_INT64,
           DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL,
           DT_COMPLEX64, DT_COMPLEX128 }))
    .INPUT(num_lower, TensorType({ DT_INT32, DT_INT64 }))
    .INPUT(num_upper, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
           DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL,
           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(MatrixBandPart)

/**
*@brief Finds unique elements in a 1D tensor. \n

*@par Inputs:
*x: 1D tensor.
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper"
are 0D scalars. \n

*@par Attributes:
*out_idx: An optional DType from: "int32, int64".
Defaults to "int32". \n

*@par Outputs:
*@li y: A Tensor. Has the same type as "x".
*@li idx: A Tensor of type "out_idx".
*@li count: A Tensor of type "out_idx". \n

*@attention Constraints:
*UniqueWithCounts runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator UniqueWithCounts.
*/

REG_OP(UniqueWithCounts)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
           DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
           DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING }))
    .OUTPUT(idx, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(count, TensorType({ DT_INT32, DT_INT64 }))
    .REQUIRED_ATTR(out_idx, Type)
    .OP_END_FACTORY_REG(UniqueWithCounts)

/**
*@brief Finds unique elements in a 1D tensor. \n

*@par Inputs:
*x: 1D tensor.
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper"
are 0D scalars. \n

*@par Attributes:
*out_idx: An optional DType from: "int32, int64". Defaults to "int32". \n

*@par Outputs:
*@li y: "x" in the unique output "y".
*@li idx: A tensor the same size as "x". The index of each value of "x". \n

*@attention Constraints:
*Unique runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Unique.
*/

REG_OP(Unique)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_idx, Type, DT_INT32)
    .OP_END_FACTORY_REG(Unique)

/**
*@brief Finds unique elements in a 1D tensor. \n

*@par Inputs:
*Input "x" is a k-dimensional tensor. Inputs "num_lower" and "num_upper"
are 0D scalars.
*Including:
* @li x: 1D tensor.
* @li axis: A Tensor of type int32. Defaults to "None". \n

*@par Attributes:
*out_idx: An optional DType from: "int32, int64".
Defaults to "int32". \n

*@par Outputs:
*@li y: "x" in the unique output "y".
*@li idx: A tensor the same size as "x". The index of each value of "x". \n

*@attention Constraints:
*UniqueExt2 runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator UniqueExt2.
*/

REG_OP(UniqueExt2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
           DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_idx, Type, DT_INT32)
    .OP_END_FACTORY_REG(UniqueExt2)

/**
*@brief Computes the inverse permutation of a tensor. \n

*@par Inputs:
*x: A k-dimensional tensor. \n

*@par Outputs:
*y: A 1D tensor. \n

*@attention Constraints:
*InvertPermutation runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator InvertPermutation.
*/

REG_OP(InvertPermutation)
    .INPUT(x, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(InvertPermutation)

/**
*@brief Checks a tensor for NaN and Inf values. \n

*@par Inputs:
*x: A k-dimensional tensor. \n

*@par Attributes:
*message: Prefix of the error message. \n

*@par Outputs:
*y: The output tensor. \n

*@attention Constraints:
*CheckNumerics runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator CheckNumerics.
*/

REG_OP(CheckNumerics)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(message, String)
    .OP_END_FACTORY_REG(CheckNumerics)

/**
*@brief Converts an array of flat indices into a tuple of coordinate arrays. \n

*@par Inputs:
*Input "indices" is a 0D or 1D tensor. Input "dims" is a 1D tensor.
* @li indices: A 0D or 1D int Tensor whose elements are indices into
the flattened version of an array of dimensions "dims".
* @li dims: A 1D int Tensor of the same type as "indices".
*The shape of the array to use for unraveling indices. \n

*@par Outputs:
*y: A Tensor. Has the same type as "indices". \n

*@attention Constraints:
*UnravelIndex runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator UnravelIndex.
*/

REG_OP(UnravelIndex)
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(dims, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(UnravelIndex)

/**
*@brief Applies upper_bound(sorted_search_values, values) along each row. \n

*@par Inputs:
*Inputs "sorted_x" and "values" are 2D tensors.
* @li sorted_x: A 2D Tensor where each row is ordered.
* @li values: A 2D Tensor with the same numbers of rows as "sorted_x. \n

*@par Attributes:
*out_type: sets the optional out_type attribute to value. \n

*@par Outputs:
*y: A Tensor with the same shape as "values". \n

*@attention Constraints:
*UpperBound runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator UpperBound.
*/

REG_OP(UpperBound)
    .INPUT(sorted_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
      DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .INPUT(values, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, \
      DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(out_type, Type)
    .OP_END_FACTORY_REG(UpperBound)

/**
*@brief Finds unique elements in a 1D tensor. \n

*@par Inputs:
*Inputs "x" and "axis" are 1D vectors.
* @li x: A 1D tensor.
* @li axis: A 1D tensor. \n

*@par Attributes:
*out_idx: An optional DType from: "int32, int64".
Defaults to "int32". \n

*@par Outputs:
*@li y: "x" in the unique output "y".
*@li idx: A tensor the same size as "x". The index of each value of "x".
*@li count: A tensor the same size as "x". The index of each value of "x". \n

*@attention Constraints:
*UniqueWithCountsExt2 runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator UniqueWithCountsExt2.
*/

REG_OP(UniqueWithCountsExt2)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING }))
    .INPUT(axis, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_STRING }))
    .OUTPUT(idx, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(count, TensorType({ DT_INT32, DT_INT64 }))
    .REQUIRED_ATTR(out_idx, Type)
    .OP_END_FACTORY_REG(UniqueWithCountsExt2)

/**
*@brief Fills the tensor with the mirror value. \n

*@par Inputs:
*Inputs "x" and "paddings" are 1D scalars.
* @li x: The tensor to be padded.
* @li paddings: A two-column matrix specifying the padding sizes.
The number of rows Has the same rank as "x". \n

*@par Attributes:
*mode: Either "REFLECT" or "SYMMETRIC". In reflect mode the padded regions
do not include the borders, while in symmetric mode the padded regions
do include the borders. \n

*@par Outputs:
*y: The padded tensor. \n

*@attention Constraints:
*MirrorPad runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MirrorPad.
*/

REG_OP(MirrorPad)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL, \
      DT_COMPLEX64, DT_COMPLEX128 }))
    .INPUT(paddings, TensorType({ DT_INT32, DT_INT64 }))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
      DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL, \
      DT_COMPLEX64, DT_COMPLEX128 }))
    .REQUIRED_ATTR(mode, String)
    .OP_END_FACTORY_REG(MirrorPad)

/**
*@brief Calculates the difference between two numbers or a list of strings. \n

*@par Inputs:
*Inputs "x" and "y" are 1D vectors.
* @li x: A Tensor. 1D. Values to keep.
* @li y: A Tensor. Must have the same type as x. 1D. Values to remove. \n

*@par Attributes:
*out_idx: An optional DType from: "int32, int64". Defaults to "int32". \n

*@par Outputs:
*@li out: A Tensor. Has the same type as "x".
*@li idx: A Tensor of type "out_idx". \n

*@attention Constraints:
*ListDiff runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ListDiff.
*/

REG_OP(ListDiff)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64}))
    .INPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64}))
    .OUTPUT(out, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
        DT_INT16, DT_UINT16, DT_INT32, DT_INT64}))
    .OUTPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .ATTR(out_idx, Type, DT_INT32)
    .OP_END_FACTORY_REG(ListDiff)

/**
*@brief Create an empty tensor, using the shape and dtype specified in attributes. \n

*@par Attributes:
*@li dtype: Specify the data type of the empty tensor.
*@li shape: Specify the shape of the empty tensor. \n

*@par Outputs:
*y: The empty constant tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator _ParallelConcatStart.
*/
REG_OP(_ParallelConcatStart)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(dtype, Type, DT_INT32)
    .ATTR(shape, ListInt, {})
    .OP_END_FACTORY_REG(_ParallelConcatStart)

/**
*@brief Creates a constant tensor from a tensor-like object. This operator is used for inference.
Operator Const has the same definition as operator Constant. \n

*@par Attributes:
*value: Required. The value and type of the resulting tensor, and no restrictions on type. \n

*@par Outputs:
*y: A constant tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Const.
*/
REG_OP(Const)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const)

/**
*@brief Creates a constant tensor for training. \n

*@par Attributes:
*value: Required. The value and type of the resulting tensor, and no restrictions on type. \n

*@par Outputs:
*y: The constant tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Const.
*/
REG_OP(Constant)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Constant)

/**
*@brief Returns a copy of the input tensor. \n

*@par Inputs:
*x: A tensor. \n

*@par Outputs:
*y: A copy of input tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Snapshot.
*/
REG_OP(Snapshot)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Snapshot)

/**
*@brief Gives a guarantee to the runtime that the input tensor is a constant. \n

*@par Inputs:
*x: A tensor. \n

*@par Outputs:
*y: The input tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator GuaranteeConst.
*/
REG_OP(GuaranteeConst)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(GuaranteeConst)

/**
*@brief Returns the target shape for broadcasting shapes "x1" and "x2". \n

*@par Inputs:
*@li x1: A tensor of type int32 or int64. A shape.
*@li x2: A tensor of the same type as "x1". The other shape. \n

*@par Outputs:
*y: A tensor. The broadcasted shape. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BroadcastArgs.
*/
REG_OP(BroadcastArgs)
    .INPUT(x1, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x2, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(BroadcastArgs)

/**
*@brief Outputs its input tensor as is and triggers an error if a gradient is requested. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*message: Will be printed in the error at the attempt to request a gradient. \n

*@par Outputs:
*y: The input tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator PreventGradient.
*/
REG_OP(PreventGradient)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(message, String, "")
    .OP_END_FACTORY_REG(PreventGradient)

/**
*@brief Returns the reduction indices for computing gradients of "x1" and "x2" with broadcast. \n

*@par Inputs:
*@li x1: A tensor of type int32 or int64.
*@li x2: A tensor of type int32 or int64.
"x2" has the same type as "x1". \n

*@par Outputs:
*@li y1: A tensor. Reduction indices of "x1".
*@li y2: A tensor. Reduction indices of "x2". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BroadcastGradientArgs.
*/
REG_OP(BroadcastGradientArgs)
    .INPUT(x1, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x2, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y1, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y2, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(BroadcastGradientArgs)

/**
*@brief Stops gradient computation. None is returned for the node where the gradient computation is stopped.


*@par Inputs:
*x: A tensor. \n

*@par Outputs:
*y: The input tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator StopGradient.
*/
REG_OP(StopGradient)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(StopGradient)

/**
*@brief Return a tensor with the same shape and contents as input. \n

*@par Inputs:
*x: A tensor. \n

*@par Outputs:
*y: A tensor with the same shape and contents as input. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Identity.
*/
REG_OP(Identity)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Identity)

/**
*@brief Returns a list of tensors with the same shapes and contents as the input tensors. \n

*@par Inputs:
*x: A list of input tensors. It's a dynamic input \n

*@par Outputs:
*y: A list of Tensor objects, with the same length as the input tensor list.
It's a dynamic output. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator IdentityN.
*/
REG_OP(IdentityN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(IdentityN)

/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*@li x: A tensor.
*@li axis: The dimension index at which to expand. \n

*@par Outputs:
*y: A tensor with the same data as input, with an additional dimension inserted at the index specified by axis. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ExpandDims.
*/
REG_OP(ExpandDims)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(ExpandDims)

/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*@li x: Original tensor.

*@par Attributes:
*@li axes: List of ints indicating the dimensions to be inserted. \n

*@par Outputs:
*y: Reshape tensor with same data as input. \n

*@par Third-party framework compatibility
*Compatible with the Onnx operator Unsqueeze.
*/

REG_OP(Unsqueeze)
    .INPUT(x, TensorType({DT_FLOAT32, DT_INT32, DT_UINT8, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT32, DT_INT32, DT_UINT8, DT_BOOL}))
    .ATTR(axes, ListInt, {})
    .OP_END_FACTORY_REG(Unsqueeze)

/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*@li x: Original tensor.

*@par Attributes:
*@li axes: List of ints indicating the dimensions to be inserted. \n

*@par Outputs:
*y: Reshape tensor with same data as input. \n

*@par Third-party framework compatibility
*Compatible with the Onnx operator Unsqueeze.

*@par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Unsqueeze instead.
*/

REG_OP(UnsqueezeV2)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(UnsqueezeV2)

/**
*@brief Reshapes a tensor. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*@li x: A tensor.
*@li shape: A tensor. Defines the shape of the output tensor. \n

*@par Attributes:
*@li axis: An optional int32 or int64. The first dimension to reshape. Defaults to "0".
*@li num_axes: An optional int32 or int64. The extent of the reshape. Defaults to "-1". \n

*@par Outputs:
*y: A tensor. \n

*@attention Constraints:
*This operator cannot be directly called by the acllopExecute API. \n

*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator Reshape.
*@li Compatible with the Caffe operator Reshape.
*/
REG_OP(Reshape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32,
        DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Reshape)

/**
*@brief Removes dimensions of size 1 from the shape of a tensor. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*axis: An optional list of int32 or int64. If not specified, squeezes all dimensions of size 1.   If specified, only squeezes the dimensions listed. It is an error to squeeze a dimension that is not 1. \n

*@par Outputs:
*y: A tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Squeeze.
*/
REG_OP(Squeeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(Squeeze)

/**
*@brief Removes dimensions of size 1 from the shape of a tensor. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*axis: An optional list of int32 or int64. If not specified, squeezes all dimensions of size 1.   If specified, only squeezes the dimensions listed. It is an error to squeeze a dimension that is not 1. \n

*@par Outputs:
*y: A tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Squeeze.

*@par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Squeeze instead.
*/
REG_OP(SqueezeV2)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(SqueezeV2)

/**
*@brief Returns an integer representing the rank of input tensor. The rank of a tensor is the number of indices required to uniquely select each element of the tensor, that is, the dimension size of the tensor. \n

*@par Inputs:
*x: A Tensor of type float32, float16, int8, int16, uint16, uint8, int32, int64, uint32, uint64, bool, double. \n

*@par Outputs:
*y: A tensor. The rank of input tensor. Type is int32. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Rank.
*/
REG_OP(Rank)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(Rank)

/**
*@brief Returns the size of a tensor, that is, an integer of the number of elements of the tensor. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*out_type: An optional int32 or int64. The output data type. Defaults to "int32". \n

*@par Outputs:
*y: A tensor. The size of the input tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Size.
*/
REG_OP(Size)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32,DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Size)

/**
*@brief Input data for other operators. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*index: Index of the input tensor.The data type must be int32 or int64.
Assume that net has three data nodes, one should be set 0, another should
be set 1, and the left should be set 2. \n

*@par Outputs:
*y: A tensor. \n

*@par Third-party framework compatibility
*Compatible with the Caffe operator Data.
*/
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

/**
*@brief Inserts a placeholder for a tensor that will be always fed. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*@li peerIndex: An integer type. The index of the corresponding "end" node connected to.
*@li parentId: A string, used to check if the nodes are from the saved parent node.
*@li parentOpType: A string. Op type of the original node.
*@li anchorIndex: An integer, used to check if the node is from the saved anchor. \n

*@par Outputs:
*y: The created placeholder tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator PlaceHolder.
*/
REG_OP(PlaceHolder)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(peerIndex, Int, 0) // the index of the corresponding 'end' node it's connected to
    .ATTR(parentId, String, "")     // check if these node are from save parent node
    .ATTR(parentOpType, String, "") // op type of original node
    .ATTR(anchorIndex, Int, 0)  // check if these node are from save anchor
    .OP_END_FACTORY_REG(PlaceHolder)

/**
*@brief Inserts a placeholder with default value for a tensor. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*@li shape: tensor shape. \n

*@par Outputs:
*y: The created placeholder tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator PlaceholderWithDefault.
*/
REG_OP(PlaceholderWithDefault)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(shape, ListInt)
    .OP_END_FACTORY_REG(PlaceholderWithDefault)

/**
*@brief Reads and returns the value of the input variable tensor. \n

*@par Inputs:
*x: A tensor must have numeric type. \n

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to int32. \n

*@par Outputs:
*y: A tensor must have numeric type. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ReadVariableOp.
*/
REG_OP(ReadVariableOp)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                          DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
                           DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(ReadVariableOp)

/**
*@brief Mark outputs of one sub graph which partitioned by engine type.

*@par Inputs:
*x: A tensor. \n

*@par Outputs:
*y: A tensor. \n

*@par Attributes:
*@li peerIndex: The index of the corresponding 'placeholder' node it's connected to.
*@li parentOpType: Op type of original node.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(End)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(peerIndex, Int, 0)
    .ATTR(parentOpType, String, "")
    .OP_END_FACTORY_REG(End)

/**
*@brief Operations for writing summary data, for use in analysis and visualization.

*@par Inputs:
* One input:
*x: Collections of summary data.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Summary)
    .INPUT(x, TensorType::ALL())
    .OP_END_FACTORY_REG(Summary)

/**
*@brief Returns the shape of a tensor. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to int32. \n

*@par Outputs:
*y: A tensor. The shape of the input tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Size.
*/
REG_OP(Shape)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Shape)

/**
*@brief Returns shape of tensors. \n

*@par Inputs:
*x: A list of input tensors. It's a dynamic input. \n

*@par Attributes:
*dtype: An optional int32 or int64. The output data type. Defaults to "int32". \n

*@par Outputs:
*y: A list of tensors with the same length as the input list of tensors.
It's a dynamic output. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ShapeN.
*/
REG_OP(ShapeN)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(ShapeN)

/**
*@brief Creates a tensor with the given "shape" and "dtype". \n

*@par Inputs:
*shape: The shape of the output tensor. \n

*@par Attributes:
*@li dtype: Optional. The data type of the output tensor. Defaults to "int32".
*@li init: An optional bool. If true, initializes the returned tensor with the default value of "dtype". Defaults to "false". \n

*@par Outputs:
*y: A tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Empty.
*/
REG_OP(Empty)
    .INPUT(shape, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8,
        DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(dtype, Int, DT_INT32)
    .ATTR(init, Bool, 0)
    .OP_END_FACTORY_REG(Empty)

/**
*@brief Gradient op for MirrorPad op. Folds a mirror-padded tensor. \n

*@par Inputs:
*Inputs "x" and "y" are 1D vectors.
* @li x: A Tensor. The input tensor to be folded.
* @li paddings: A Tensor of type int32 or int64. A two-column matrix
specifying the padding sizes. \n

*@par Attributes:
*mode: A string from: "REFLECT", "SYMMETRIC". The mode used in the MirrorPad op. \n

*@par Outputs:
*y: A Tensor. Has the same type as "x". \n

*@attention Constraints:
*MirrorPadGrad runs on the Ascend AI CPU, which delivers poor performance. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator MirrorPadGrad.
*/

REG_OP(MirrorPadGrad)
    .INPUT(x, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
              DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
              DT_COMPLEX64, DT_COMPLEX128 }))
    .INPUT(paddings, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({ DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
              DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, \
              DT_COMPLEX64, DT_COMPLEX128 }))
    .REQUIRED_ATTR(mode, String)
    .OP_END_FACTORY_REG(MirrorPadGrad)

/**
*@brief Returns locations of nonzero / true values in a tensor. \n

*@par Inputs:
*Including:
*x: A Tensor. Must be one of the following types:
DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16,
DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL. \n

*@par Outputs:
*y: A Tensor of type DT_INT64. \n

*@attention Constraints:
*Where runs on the Ascend AI CPU, which delivers poor performance.\n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Where.
*/

REG_OP(Where)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \
              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(Where)

/**
*@brief Derived from the Caffe operator Split that splits an input blob to
*    multiple output blobs for feeding a blob into multiple output layers.
*The Split node is removed from the graph after the split operation is completed. \n

*@par Inputs:
*x: A Tensor. Must be one of the following types:
fp16, fp32, int8, uint8, int16, uint16, int32, uint32, int64, uint64. \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".It's required and the value should equal to output_num. \n

*@par Attributes:
*@li N: A required int. The parameter will get the number of dynamic outputs.
*/
REG_OP(Copy)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \
              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \
              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64}))
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(Copy);

/**
*@brief Generates fingerprint values. \n

*@par Inputs:
*@li data: Must have rank 1 or higher.
*@li method: Fingerprint method used by this op. Currently available method is
`farmhash::fingerprint64`. \n

*@par Outputs:
y: A two-dimensional `Tensor` of type `tf.uint8`. The first dimension equals to
`data`'s first dimension, and the second dimension size depends on the
fingerprint algorithm. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow Fingerprint operator.
*/

REG_OP(Fingerprint)
    .INPUT(data, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \
              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .INPUT(method, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(Fingerprint)

/**
*@brief Change the shape of output according to the attr outShape
*

*@par Inputs:
*x: A Tensor. \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".It's required and the value should equal to output_num. \n

*@par Attributes:
*outShape: The shape of output will be inferred according to the attribute
*/
REG_OP(TransShape)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(outShape,ListInt ,{})
    .OP_END_FACTORY_REG(TransShape);

/**
*@brief Computes the (possibly normalized) Levenshtein Edit Distance. \n

*@par Inputs:
*@li hypothesis_indices: The indices of the hypothesis list SparseTensor.
This is an N x R int64 matrix.
*@li hypothesis_shape: The values of the hypothesis list SparseTensor.
This is an N-length vector.
*@li hypothesis_shape: The shape of the hypothesis list SparseTensor.
This is an R-length vector.
*@li truth_indices: The indices of the truth list SparseTensor.
This is an M x R int64 matrix.
*@li truth_shape: The values of the truth list SparseTensor.
This is an M-length vector.
*@li truth_shape: The shape of the truth list SparseTensor.
This is an R-length vector

*@par Attributes:
*normalize: boolean (if true, edit distances are normalized by length of truth). \n

*@par Outputs:
*output: A dense float tensor with rank R - 1. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow EditDistance operator.
*/
REG_OP(EditDistance)
    .INPUT(hypothesis_indices, TensorType({DT_INT64}))
    .INPUT(hypothesis_values, TensorType::BasicType())
    .INPUT(hypothesis_shape, TensorType({DT_INT64}))
    .INPUT(truth_indices, TensorType({DT_INT64}))
    .INPUT(truth_values, TensorType::BasicType())
    .INPUT(truth_shape, TensorType({DT_INT64}))
    .ATTR(normalize, Bool, true)
    .OUTPUT(output, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(EditDistance)

/**
* @brief sort the input tensor without returning the value of index.

* @par Inputs:
* x: An ND tensor of type float16.

* @par Attributes:
* @li axis: An optional int. The dimension to sort along. This value defaults to -1.
* @li descending: An optional bool. Controls the sorting order (ascending or descending). This value defaults to False.

* @par Outputs:
* y: An ND tensor of type float16.

* @attention Constraints:
* @li Axis should select the last dim.
* @li When the sorting data is less than 150K, it is recommended to use this tbe ops,
 and the descending performance is better than the ascending.
* @li The upper limit of data on Ascend910 is 2000K.
*/
REG_OP(SortV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(axis, Int, -1)
    .ATTR(descending, Bool, false)
    .OP_END_FACTORY_REG(SortV2)

/**
* @brief Expand the input tensor to a compatible shape. \n

* @par Inputs:
* One inputs, including:
* @li x: A Tensor. Must be one of the following types:
*     float16, float32, int32, int8 ,uint8. \n
* @li shape: A Tensor to specify the shape that the input tensor expanded to. \n

* @par Outputs:
* @li y: A Tensor. Has the same type as "x", and the shape specified by input and attr shape \n

* @par Third-party framework compatibility
* Compatible with the ONNX operator Expand.
*/

REG_OP(Expand)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
    .INPUT(shape, TensorType({DT_INT16, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
    .OP_END_FACTORY_REG(Expand)

/**
*@Returns a tensor containing the indices of all non-zero elements of input. \n

*@par Inputs:
*x: A Tensor. Must be one of the following types: float16, float32, int32, int64.

*@par Attributes:
* transpose: the output tensor will be transposed if true. \n

*@par Outputs:
* y: A Tensor. Has the same type as "x" . \n

*@par Third-party framework compatibility
*Compatible with the PyTorch operator NonZero.
*/

REG_OP(NonZero)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \
              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT64, DT_INT32}))
    .ATTR(transpose, Bool, false)
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(NonZero)

/**
* @brief Expand the input tensor to a compatible shape. \n

* @par Inputs:
* One inputs, including:
* x: A Tensor. Must be one of the following types:
*     float16, float32, int32, int8 ,uint8. \n

* @par Attributes:
* shape: A required listInt to specify the shape that the input tensor expanded to. \n


* @par Outputs:
* y: A Tensor. Has the same type as "x", and the shape specified by input and attr shape \n

* @par Third-party framework compatibility
* Compatible with the ONNX operator Expand.
*/

REG_OP(ExpandD)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8}))
    .REQUIRED_ATTR(shape, ListInt)
    .OP_END_FACTORY_REG(ExpandD)

/**
* @brief Calculate buckets limit and offset. \n

* @par Inputs:
* Three inputs, including:
* @li bucket_list: A 1-D tensor of type int32 with the value of ivf_counts and ivf_offset index. \n
* @li ivf_counts: A 1-D tensor of type int32 with the value of ivf counts. \n
* @li ivf_offset: A 1-D tensor of type int32 or int64 with the value of ivf offset. \n

* @par Attributes:
* total_limit: A int64 type maximum value of the sum of ivf_counts corresponding to bucket_list. \n

* @par Outputs:
* @li buckets_limit: A 1-D tensor of type int32 with the sum <= total_limit. \n
* @li buckets_offset: A 1-D tensor of type int32 or int64 with the value of ivf_offset corresponding to bucket_list. \n
*/
REG_OP(CalcBucketsLimitAndOffset)
    .INPUT(bucket_list, TensorType({DT_INT32}))
    .INPUT(ivf_counts, TensorType({DT_INT32}))
    .INPUT(ivf_offset, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(buckets_limit, TensorType({DT_INT32}))
    .OUTPUT(buckets_offset, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(total_limit, Int)
    .OP_END_FACTORY_REG(CalcBucketsLimitAndOffset)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_ARRAY_OPS_H_
