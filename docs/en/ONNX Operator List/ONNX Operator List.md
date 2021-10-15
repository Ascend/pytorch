# ONNX Operator List
-   [Abs](#abs.md)
-   [Acos](#acos.md)
-   [Acosh](#acosh.md)
-   [AdaptiveAvgPool2D](#adaptiveavgpool2d.md)
-   [AdaptiveMaxPool2D](#adaptivemaxpool2d.md)
-   [Add](#add.md)
-   [Addcmul](#addcmul.md)
-   [AffineGrid](#affinegrid.md)
-   [And](#and.md)
-   [Argmax](#argmax.md)
-   [Argmin](#argmin.md)
-   [AscendRequantS16](#ascendrequants16.md)
-   [AscendRequant](#ascendrequant.md)
-   [AscendQuant](#ascendquant.md)
-   [AscendDequantS16](#ascenddequants16.md)
-   [AscendDequant](#ascenddequant.md)
-   [AscendAntiQuant](#ascendantiquant.md)
-   [Asin](#asin.md)
-   [Asinh](#asinh.md)
-   [Atan](#atan.md)
-   [Atanh](#atanh.md)
-   [AveragePool](#averagepool.md)
-   [BatchNormalization](#batchnormalization.md)
-   [BatchMatMul](#batchmatmul.md)
-   [BatchMultiClassNMS](#batchmulticlassnms.md)
-   [BitShift](#bitshift.md)
-   [Cast](#cast.md)
-   [Ceil](#ceil.md)
-   [Celu](#celu.md)
-   [Concat](#concat.md)
-   [Clip](#clip.md)
-   [ConvTranspose](#convtranspose.md)
-   [Cumsum](#cumsum.md)
-   [Conv](#conv.md)
-   [Compress](#compress.md)
-   [Constant](#constant.md)
-   [ConstantOfShape](#constantofshape.md)
-   [Cos](#cos.md)
-   [Cosh](#cosh.md)
-   [DeformableConv2D](#deformableconv2d.md)
-   [Det](#det.md)
-   [DepthToSpace](#depthtospace.md)
-   [Div](#div.md)
-   [Dropout](#dropout.md)
-   [Elu](#elu.md)
-   [EmbeddingBag](#embeddingbag.md)
-   [Equal](#equal.md)
-   [Erf](#erf.md)
-   [Exp](#exp.md)
-   [Expand](#expand.md)
-   [EyeLike](#eyelike.md)
-   [Flatten](#flatten.md)
-   [Floor](#floor.md)
-   [Gather](#gather.md)
-   [GatherND](#gathernd.md)
-   [GatherElements](#gatherelements.md)
-   [Gemm](#gemm.md)
-   [GlobalAveragePool](#globalaveragepool.md)
-   [GlobalLpPool](#globallppool.md)
-   [GlobalMaxPool](#globalmaxpool.md)
-   [Greater](#greater.md)
-   [GreaterOrEqual](#greaterorequal.md)
-   [HardSigmoid](#hardsigmoid.md)
-   [hardmax](#hardmax.md)
-   [HardSwish](#hardswish.md)
-   [Identity](#identity.md)
-   [If](#if.md)
-   [InstanceNormalization](#instancenormalization.md)
-   [Less](#less.md)
-   [LeakyRelu](#leakyrelu.md)
-   [LessOrEqual](#lessorequal.md)
-   [Log](#log.md)
-   [LogSoftMax](#logsoftmax.md)
-   [LpNormalization](#lpnormalization.md)
-   [LpPool](#lppool.md)
-   [LRN](#lrn.md)
-   [LSTM](#lstm.md)
-   [MatMul](#matmul.md)
-   [Max](#max.md)
-   [MaxPool](#maxpool.md)
-   [MaxRoiPool](#maxroipool.md)
-   [MaxUnpool](#maxunpool.md)
-   [Mean](#mean.md)
-   [MeanVarianceNormalization](#meanvariancenormalization.md)
-   [Min](#min.md)
-   [Mod](#mod.md)
-   [Mul](#mul.md)
-   [Multinomial](#multinomial.md)
-   [Neg](#neg.md)
-   [NonMaxSuppression](#nonmaxsuppression.md)
-   [NonZero](#nonzero.md)
-   [Not](#not.md)
-   [OneHot](#onehot.md)
-   [Or](#or.md)
-   [RandomNormalLike](#randomnormallike.md)
-   [RandomUniformLike](#randomuniformlike.md)
-   [RandomUniform](#randomuniform.md)
-   [Range](#range.md)
-   [Reciprocal](#reciprocal.md)
-   [ReduceL1](#reducel1.md)
-   [ReduceL2](#reducel2.md)
-   [ReduceLogSum](#reducelogsum.md)
-   [ReduceLogSumExp](#reducelogsumexp.md)
-   [ReduceMin](#reducemin.md)
-   [ReduceMean](#reducemean.md)
-   [ReduceProd](#reduceprod.md)
-   [ReduceSumSquare](#reducesumsquare.md)
-   [Resize](#resize.md)
-   [Relu](#relu.md)
-   [ReduceSum](#reducesum.md)
-   [ReduceMax](#reducemax.md)
-   [Reshape](#reshape.md)
-   [ReverseSequence](#reversesequence.md)
-   [RoiExtractor](#roiextractor.md)
-   [RoiAlign](#roialign.md)
-   [Round](#round.md)
-   [PRelu](#prelu.md)
-   [Scatter](#scatter.md)
-   [ScatterElements](#scatterelements.md)
-   [ScatterND](#scatternd.md)
-   [Shrink](#shrink.md)
-   [Selu](#selu.md)
-   [Shape](#shape.md)
-   [Sigmoid](#sigmoid.md)
-   [Slice](#slice.md)
-   [Softmax](#softmax.md)
-   [Softsign](#softsign.md)
-   [Softplus](#softplus.md)
-   [SpaceToDepth](#spacetodepth.md)
-   [Split](#split.md)
-   [Sqrt](#sqrt.md)
-   [Squeeze](#squeeze.md)
-   [Sub](#sub.md)
-   [Sign](#sign.md)
-   [Sin](#sin.md)
-   [Sinh](#sinh.md)
-   [Size](#size.md)
-   [Sum](#sum.md)
-   [Tanh](#tanh.md)
-   [TfIdfVectorizer](#tfidfvectorizer.md)
-   [Tile](#tile.md)
-   [ThresholdedRelu](#thresholdedrelu.md)
-   [TopK](#topk.md)
-   [Transpose](#transpose.md)
-   [Pad](#pad.md)
-   [Pow](#pow.md)
-   [Unsqueeze](#unsqueeze.md)
-   [Xor](#xor.md)
-   [Where](#where.md)
<h2 id="abs.md">Abs</h2>

## Description<a name="section12725193815114"></a>

Computes the absolute value of a tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor. Must be one of the following types: float16, float32, double, int32, int64.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="acos.md">Acos</h2>

## Description<a name="section12725193815114"></a>

Computes acos of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="acosh.md">Acosh</h2>

## Description<a name="section12725193815114"></a>

Computes inverse hyperbolic cosine of x element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="adaptiveavgpool2d.md">AdaptiveAvgPool2D</h2>

## Description<a name="section12725193815114"></a>

Applies a 2D adaptive avg pooling over the input.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Attributes\]

One attribute

output\_size: array of ints, specifying the output H and W dimension sizes.

\[Outputs\]

One output

y: tensor of the identical data type as x.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="adaptivemaxpool2d.md">AdaptiveMaxPool2D</h2>

## Description<a name="section12725193815114"></a>

Applies a 2D adaptive max pooling over the input.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or float64.

\[Attributes\]

One attribute

output\_size: array of ints, specifying the output H and W dimension sizes.

\[Outputs\]

Two outputs

y: tensor of the identical data type as x.

argmax: tensor of type int32 or int64.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="add.md">Add</h2>

## Description<a name="section12725193815114"></a>

Adds inputs element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

A: tensor. Must be one of the following types: int8, int16, int32, int64, uint8, float32, float16, double.

B: tensor of the identical data type as A.

\[Outputs\]

C: tensor of the identical data type as A.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="addcmul.md">Addcmul</h2>

## Description<a name="section12725193815114"></a>

Performs element-wise computation: \(x1 \* x2\) \* value + input\_data

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Four inputs

input\_data: tensor of type float16, float32, int32, int8, or uint8.

x1: tensor of the identical data type as input\_data

x2: tensor of the identical data type as input\_data

value: tensor of the identical data type as input\_data

\[Outputs\]

One output

y: tensor of the identical data type as the inputs.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="affinegrid.md">AffineGrid</h2>

## Description<a name="section12725193815114"></a>

Generates a sampling grid with given matrices.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

theta: tensor of type float16 or float32.

output\_size: tensor of type int32.

\[Attributes\]

One attribute

align\_corners: bool

\[Outputs\]

One output

y: tensor of type int.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="and.md">And</h2>

## Description<a name="section12725193815114"></a>

Returns the tensor resulted from performing the and logical operation element-wise on the input tensors.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x1: tensor of type bool.

x2: tensor of type bool.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="argmax.md">Argmax</h2>

## Description<a name="section12725193815114"></a>

Returns the indices of the maximum elements along the provided axis.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of type int32, the indexes. Has the same shape as x with the dimension along axis removed.

\[Attributes\]

axis: \(required\) int32, axis in which to compute the arg indices. Accepted range is \[–len\(x.shape\), len\(x.shape\) – 1\].

keep\_dim: \(optional\) either 1 \(default\) or 0.

\[Restrictions\]

The operator does not support inputs of type float32 when the atc command-line option  **--precision\_mode**  is set to  **must\_keep\_origin\_dtype**.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="argmin.md">Argmin</h2>

## Description<a name="section12725193815114"></a>

Returns the indices of the minimum values along an axis.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of type int64.

\[Attributes\]

axis: int. Must be in the range \[–r, r – 1\], where r indicates the rank of the input.

\[Restrictions\]

The operator does not support inputs of type float32 when the atc command-line option  **--precision\_mode**  is set to  **must\_keep\_origin\_dtype**.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ascendrequants16.md">AscendRequantS16</h2>

## Description<a name="section12725193815114"></a>

Performs requantization.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two required inputs and one optional input

x0: tensor of type int16.

req\_scale: tensor of type uint64.

x1: tensor of type int16.

\[Attributes\]

Two attributes

dual\_output: bool

relu\_flag: bool

\[Outputs\]

Two outputs

y0: tensor of type int8.

y1: tensor of type int16.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="ascendrequant.md">AscendRequant</h2>

## Description<a name="section12725193815114"></a>

Performs requantization.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x0: tensor of type int32.

req\_scale: tensor of type uint64.

\[Attributes\]

One attribute

relu\_flag: bool

\[Outputs\]

One output

y: tensor of type int8.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="ascendquant.md">AscendQuant</h2>

## Description<a name="section12725193815114"></a>

Performs quantization.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Attributes\]

Four attributes

offset: float

scale: float

sqrt\_mode: bool

round\_mode: string

\[Outputs\]

One output

y: tensor of type int8.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="ascenddequants16.md">AscendDequantS16</h2>

## Description<a name="section12725193815114"></a>

Performs dequantization.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two required inputs and one optional input

x0: tensor of type int32.

req\_scale: tensor of type uint64.

x1: tensor of type int16.

\[Attributes\]

One attribute

relu\_flag: bool

\[Outputs\]

One output

y: tensor of type int16.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="ascenddequant.md">AscendDequant</h2>

## Description<a name="section12725193815114"></a>

Performs dequantization.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x0: tensor of type int32.

deq\_scale: tensor of type uint64 or float16.

\[Attributes\]

sqrt\_mode: bool

relu\_flag: bool

dtype: float

\[Outputs\]

One output

y: tensor of type float16 or float.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="ascendantiquant.md">AscendAntiQuant</h2>

## Description<a name="section12725193815114"></a>

Performs dequantization.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type int8.

\[Attributes\]

offset: float

scale: float

sqrt\_mode: bool

round\_mode: string

\[Outputs\]

One output

y: tensor of type float16 or float.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="asin.md">Asin</h2>

## Description<a name="section12725193815114"></a>

Computes the trignometric inverse sine of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x1: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="asinh.md">Asinh</h2>

## Description<a name="section12725193815114"></a>

Computes inverse hyperbolic sine of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="atan.md">Atan</h2>

## Description<a name="section12725193815114"></a>

Computes the trignometric inverse tangent of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="atanh.md">Atanh</h2>

## Description<a name="section12725193815114"></a>

Computes inverse hyperbolic tangent of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="averagepool.md">AveragePool</h2>

## Description<a name="section12725193815114"></a>

Performs average pooling.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

X: tensor of type float16 or float32, in NCHW format.

\[Outputs\]

Y: tensor of type float16 or float32, in NCHW format.

\[Attributes\]

auto\_pad: \(optional\) selected from NOTSET, SAME\_UPPER, SAME\_LOWER, and VALID.

count\_include\_pad: int, not supported currently.

kernel\_shape: \(optional\)

-   kernel\_shape\[0\]: int32, the kernel height. Must be in the range \[1, 32768\]. Defaults to 1.

-   kernel\_shape\[1\]: int32, the kernel width. Must be in the range \[1, 32768\]. Defaults to 1.

strides: \(optional\)

-   strides\[0\]: int32, the stride height. Defaults to 1.

-   strides\[1\]: int32, the stride width. Defaults to 1.

pads: \(optional\)

-   pads\[0\]: int32, top padding. Defaults to 0.

-   pads\[1\]: int32, bottom padding. Defaults to 0.

-   pads\[2\]: int32, left padding. Defaults to 0.

-   pads\[3\]: int32, right padding. Defaults to 0.

ceil\_mode: \(optional\) int32, either 0 \(floor mode\) or 1 \(ceil mode\). Defaults to 0.

\[Restrictions\]

When strides\[0\] or strides\[1\] is greater than 63, computation is performed on AI CPU, which will compromise performance.

When the value of kernel\_shape\_H or kernel\_shape\_W is beyond the range \[1,255\] or kernel\_shape\_H \* kernel\_shape\_W \> 256, computation is performed on AI CPU, which will compromise performance.

input\_w ∈ \[1, 4096\]

When N of the input tensor is a prime number, N < 65535.

ceil\_mode is valid only when auto\_pad is set to NOTSET.

The operator does not support inputs of type float32 when the atc command-line option  **--precision\_mode**  is set to  **must\_keep\_origin\_dtype**.

Beware that both the SAME\_UPPER and SAME\_LOWER values of auto\_pad are functionally the same as the SAME argument of built-in TBE operators. The attribute configuration may lead to accuracy drop as the SAME argument is position-insensitive.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="batchnormalization.md">BatchNormalization</h2>

## Description<a name="section12725193815114"></a>

Normalizes the inputs.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Five inputs

X: 4D tensor of type float16 or float32.

scale: tensor of type float32, specifying the scale factor.

B: tensor of type float32, specifying the offset.

mean: tensor of type float32, specifying the mean value.

var: tensor of type float32, specifying the variance value.

\[Outputs\]

Five outputs

Y: normalized tensor of type float16 or float32.

mean: mean value.

var: variance value.

saved\_mean: saved mean value, used to accelerate gradient calculation during training.

saved\_var: saved variance value, used to accelerate gradient calculation during training.

\[Attributes\]

epsilon: \(optional\) float32, added to var to avoid dividing by zero. Defaults to 0.0001.

momentum: float32, not supported currently.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="batchmatmul.md">BatchMatMul</h2>

## Description<a name="section12725193815114"></a>

Multiplies slices of two tensors in batches.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x1: tensor of type float16, float, or int32.

x2: tensor of type float16, float, or int32.

\[Attributes\]

Two attributes

adj\_x1: bool

adj\_x2: bool

\[Outputs\]

One output

y: tensor of type float16, float, or int32.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="batchmulticlassnms.md">BatchMultiClassNMS</h2>

## Description<a name="section12725193815114"></a>

Applies non-maximum suppression \(NMS\) on input boxes and input scores.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two required inputs and two optional inputs

boxes: tensor of type float16

scores: tensor of type float16

clip\_window: tensor of type float16

num\_valid\_boxes: tensor of type int32

\[Attributes\]

Six attributes

score\_threshold: float

iou\_threshold: float

max\_size\_per\_class: int

max\_total\_size: int

change\_coordinate\_frame: bool

transpose\_box: bool

\[Outputs\]

Four outputs

nmsed\_boxes: tensor of type float16

nmsed\_scores: tensor of type float16

nmsed\_classes: tensor of type float16

nmsed\_num: tensor of type float16

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="bitshift.md">BitShift</h2>

## Description<a name="section421532641316"></a>

Performs element-wise shift.

## Parameters<a name="section143631030111310"></a>

\[Inputs\]

Two inputs

x: tensor, indicating the input to be shifted.

y: tensor, indicating the amounts of shift.

\[Outputs\]

z: shifted tensor.

\[Attributes\]

direction: \(required\) string, indicating the direction of moving bits. Either RIGHT or LEFT.

\[Restrictions\]

When direction="LEFT", the inputs must not be of type UINT16, UIN32, or UINT64.

## ONNX Opset Support<a name="section098583811132"></a>

Opset v11/v12/v13

<h2 id="cast.md">Cast</h2>

## Description<a name="section12725193815114"></a>

Casts a tensor to a new type.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor

\[Outputs\]

y: tensor of the data type specified by the attribute. Must be one of the following types: bool, float16, float32, int8, int32, uint8.

\[Attributes\]

to: \(required\) int, the destination type.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ceil.md">Ceil</h2>

## Description<a name="section12725193815114"></a>

Returns the ceiling of the input, element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="celu.md">Celu</h2>

## Description<a name="section12725193815114"></a>

Continuously Differentiable Exponential Linear Units \(CELUs\): performs the linear unit element-wise on the input tensor X using formula:

max\(0,x\) + min\(0,alpha \* \(exp\(x/alpha\) – 1\)\)

## Parameters<a name="section9981612134"></a>

\[Inputs\]

X: tensor of type float.

\[Outputs\]

Y: tensor of type float.

\[Attributes\]

alpha: float. Defaults to 1.0.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v12/v13

<h2 id="concat.md">Concat</h2>

## Description<a name="section12725193815114"></a>

Concatenates multiple inputs.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

inputs: tensors. Must be one of the following data types: float16, float32, int32, uint8, int16, int8, int64, qint8, quint8, qint32, uint16, uint32, uint64, qint16, quint16.

\[Outputs\]

concat\_result: tensor of the identical data type as inputs.

\[Attributes\]

axis: the axis along which to concatenate — may be negative to index from the end. Must be in the range \[–r, r – 1\], where, r = rank\(inputs\).

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="clip.md">Clip</h2>

## Description<a name="section12725193815114"></a>

Clips tensor values to a specified min and max.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Three inputs

X: tensor of type float16, float32, or int32.

min: must be a scalar.

max: must be a scalar.

\[Outputs\]

One output

Y: output tensor with clipped input elements. Has the identical shape and data type as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="convtranspose.md">ConvTranspose</h2>

## Description<a name="section12725193815114"></a>

Computes transposed convolution.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Three inputs

x: tensor of type float16 or float32.

w: tensor of type float16 or float32.

b: \(optional\) tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Attributes\]

auto\_pad: string. Defaults to NOTSET, which means explicit padding is used.

dilations: ints. Dilation value along each spatial axis of the filter. Defaults to 1, meaning along each spatial axis.

group: int. Number of groups input channels and output channels are divided into. Defaults to 1.

kernel\_shape: ints. The shape of the convolution kernel. Defaults to w.

output\_padding: ints. Additional elements added to the side with higher coordinate indices in the output. Defaults to an all-0 array.

output\_shape: ints. The shape of the output can be explicitly set which will cause pads values to be auto generated.

pads: ints. Padding for the beginning and ending along each spatial axis. Defaults to an all-0 matrix.

strides: ints. Stride along each spatial axis. Defaults to an all-1 matrix.

\[Restrictions\]

Currently, only 2D transposed convolution is supported. 3D and higher are not supported.

dilations can only be 1.

Currently, the output\_shape can be used to specify the output shape size. But the specified size must not be greater than the input size.

The operator does not support inputs of type float32 or float64 when the atc command-line option  **--precision\_mode**  is set to  **must\_keep\_origin\_dtype**.

The auto\_pad attribute must not be SAME\_UPPER or SAME\_LOWER.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="cumsum.md">Cumsum</h2>

## Description<a name="section12725193815114"></a>

Performs cumulative sum of the input elements along the given axis.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x: tensor of type float16, float32, or int32.

axis: scalar of type int32 or int64. Defaults to 0. Must be in the range \[–rank\(x\), rank\(x\) – 1\].

\[Outputs\]

One output

y: tensor of the identical data type as input x.

\[Attributes\]

exclusive: int. Whether to return exclusive sum in which the top element is not included. Defaults to 0.

reverse: int. Whether to perform the sums in reverse direction. Defaults to 0.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="conv.md">Conv</h2>

## Description<a name="section12725193815114"></a>

Computes convolution.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

X: 4D tensor

W: tensor for the weight

B: \(optional\) 1D tensor for the bias

\[Outputs\]

Y: tensor for the convolution output

\[Attributes\]

auto\_pad: \(optional\) either VALID or NOTSET.

dilations: list of four integers, specifying the dilation rate. The value range for the H and W dimensions is \[1, 255\].

group: int32. The input and output channels are separated into groups, and the output group channels will be only connected to the input group channels. Both the input and output channels must be divisible by group. Must be 1.

pads: list of four integers, specifying the number of pixels to add to each side of the input. Must be in the range \[0, 255\].

strides: list of four integers, specifying the strides of the convolution along the height and width. The value range for the H and W dimensions is \[1, 63\]. By default, the N and C dimensions are set to 1.

\[Restrictions\]

For input X, the value range for the W dimension is \[1, 4096\].

For the weight tensor, the value range for the H and W dimensions is \[1, 255\].

When W and H of the output tensor are both 1, inputs X and W must have the same H and W dimensions.

The operator is not supported if the output Y meets: W = 1, H ! = 1

The operator does not support inputs of type float32 or float64 when the atc command-line option  **--precision\_mode**  is set to  **must\_keep\_origin\_dtype**.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="compress.md">Compress</h2>

## Description<a name="section12725193815114"></a>

Slices data based on the specified axis.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs:

input: tensor with one or more dimensions. The supported types are uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float, string, and bool.

condition: 1-dimensional tensor, used to specify slices and elements to be selected. The supported type is bool.

\[Outputs\]

One output

output: tensor of the same type as the input

\[Attributes\]

\(Optional\) axis: int, axis for slicing. If no axis is specified, the input tensor is flattened before slicing. The value range is \[-r, r-1\].  **r**  indicates the dimensions of the input tensor.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v9//v11/v12/v13

<h2 id="constant.md">Constant</h2>

## Description<a name="section12725193815114"></a>

Creates a constant tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

N/A

\[Outputs\]

One output

Y: output tensor containing the same value of the provided tensor.

\[Attributes\]

value: the value for the elements of the output tensor.

\[Restrictions\]

sparse\_value: not supported

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="constantofshape.md">ConstantOfShape</h2>

## Description<a name="section12725193815114"></a>

Generates a tensor with given value and shape.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

x: 1D tensor of type int64, the shape of the output tensor. All values must be greater than 0.

\[Outputs\]

y: output tensor of shape specified by the input. If value is specified, the value and data type of the output tensor is taken from value. If value is not specified, the value in the output defaults to 0, and the data type defaults to float32.

\[Attributes\]

value: the value and data type of the output elements.

\[Restrictions\]

x: 1 <= len\(shape\) <= 8

## ONNX Opset Support<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="cos.md">Cos</h2>

## Description<a name="section12725193815114"></a>

Computes cos of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="cosh.md">Cosh</h2>

## Description<a name="section12725193815114"></a>

Computes hyperbolic cosine of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

X1: tensor of type float16, float, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="deformableconv2d.md">DeformableConv2D</h2>

## Description<a name="section421532641316"></a>

Deformable convolution

## Parameters<a name="section143631030111310"></a>

\[Inputs\]

X: 4D tensor

filter: weight tensor

offsets: 4D tensor for the offset

bias: \(optional\) 1D tensor for the bias

\[Outputs\]

Y: deformed tensor

\[Attributes\]

auto\_pad: \(optional\) either VALID or NOTSET.

dilations: list of four integers, specifying the dilation rate. The value range for the H and W dimensions is \[1, 255\].

group: int32. The input and output channels are separated into groups, and the output group channels will be only connected to the input group channels. Both the input and output channels must be divisible by group. Must be 1.

pads: list of four integers, specifying the number of pixels to add to each side of the input. Must be in the range \[0, 255\].

strides: list of four integers, specifying the strides of the convolution along the height and width. The value range for the H and W dimensions is \[1, 63\]. By default, the N and C dimensions are set to 1.

data\_format: string, specifying the format of the input data. Defaults to NHWC.

deformable\_groups: the number of deformable group partitions. Defaults to 1.

modulated: bool to specify the DeformableConv2D version. Set to true to use v2; set to false to use v1. Currently, only true \(v2\) is supported.

Restrictions

For the input tensor X, expected range of the W dimension is \[1, 4096/filter\_width\] and expected range of the H dimension is \[1, 100000/filter\_height\].

For the weight tensor, expected range of both the W and H dimensions are \[1, 63\].

The operator does not support inputs of type float32 or float64 when the atc command-line option  **--precision\_mode**  is set to  **must\_keep\_origin\_dtype**.

## ONNX Opset Support<a name="section19647924181413"></a>

No ONNX support for this custom operator

<h2 id="det.md">Det</h2>

## Description<a name="section12725193815114"></a>

Calculates determinant of a square matrix or batches of square matrices.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="depthtospace.md">DepthToSpace</h2>

## Description<a name="section12725193815114"></a>

Rearranges \(permutes\) data from depth into blocks of spatial data.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

input: input tensor in format NCHW. Must be one of the following types: float16, float32, double, int32, int64.

\[Outputs\]

One output

output: tensor with shape \[N, C/\(blocksize \* blocksize\), H \* blocksize, W \* blocksize\]

\[Attributes\]

blocksize: \(required\) int, blocks to be moved.

mode: string, either DCR \(default\) for depth-column-row order re-arrangement or CRD for column-row-depth order arrangement.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="div.md">Div</h2>

## Description<a name="section12725193815114"></a>

Performs element-wise division.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x1: tensor of type float16, float32, double, int32, or int64.

x2: tensor of type float16, float32, double, int32, or int64.

\[Outputs\]

One output

y: tensor of the identical data type as the inputs.

\[Restrictions\]

The output has the identical data type as the inputs.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="dropout.md">Dropout</h2>

## Description<a name="section12725193815114"></a>

Copies or masks the input tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One to three inputs

data: input tensor, of type float16, float32, or double.

ratio: \(optional\) float16, float32, or double.

training\_mode: \(optional\) bool

\[Outputs\]

One to two outputs

output: tensor

mask: tensor

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="elu.md">Elu</h2>

## Description<a name="section12725193815114"></a>

Computes the exponential linear function.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the same data type and shape as input x.

\[Attributes\]

alpha: float, indicating the coefficient. Defaults to 1.0.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="embeddingbag.md">EmbeddingBag</h2>

## Description<a name="section12725193815114"></a>

Computes sums, means, or maxes of bags of embeddings.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two required inputs and two optional inputs

weight: tensor of type float32.

indices: tensor of type int32.

offset: tensor of type int32.

per\_sample\_weights: tensor of type float32.

\[Attributes\]

Four attributes

mode: string

scale\_grad\_by\_fraq: bool

sparse: bool

include\_last\_offset: bool

\[Outputs\]

One output

y: tensor of type float32.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="equal.md">Equal</h2>

## Description<a name="section12725193815114"></a>

Returns the truth value of \(X1 == X2\) element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

X1: tensor

X2: tensor

\[Outputs\]

One output

y: tensor of type bool.

\[Restrictions\]

X1 and X2 have the same format and data type. The following data types are supported: bool, uint8, int8, int16, int32, int64, float16, float32, and double.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="erf.md">Erf</h2>

## Description<a name="section12725193815114"></a>

Computes the Gauss error function of x element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor. Has the identical data type and format as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="exp.md">Exp</h2>

## Description<a name="section12725193815114"></a>

Computes exponential of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="expand.md">Expand</h2>

## Description<a name="section12725193815114"></a>

Broadcasts the input tensor following the given shape and the broadcast rule.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

input: tensor of type float16 or float32.

shape: tensor of type int64.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Restrictions\]

The model's inputs need to be changed from placeholders to constants. You can use ONNX Simplifier to simplify your model.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="eyelike.md">EyeLike</h2>

## Description<a name="section421532641316"></a>

Generate a 2D tensor \(matrix\) with ones on the diagonal and zeros everywhere else.

## Parameters<a name="section143631030111310"></a>

\[Inputs\]

One input

x: 2D tensor, to be copied.

\[Outputs\]

One output

y: tensor of the identical shape as input x.

\[Attributes\]

dtype: int, specifying the data type of the output.

k: int, specifying the index of the diagonal to be populated with ones. Defaults to 0. If y is output, y\[i, i+k\] = 1.

\[Restrictions\]

k must be 0.

## ONNX Opset Support<a name="section19647924181413"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="flatten.md">Flatten</h2>

## Description<a name="section12725193815114"></a>

Flattens the input.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

input: ND tensor. Must be one of the following data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32.

\[Outputs\]

2D tensor with the content of the input tensor.

\[Attributes\]

axis: int. Must be positive.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="floor.md">Floor</h2>

## Description<a name="section12725193815114"></a>

Returns element-wise largest integer not greater than x.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="gather.md">Gather</h2>

## Description<a name="section12725193815114"></a>

Gathers slices from the input according to indices.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x1: tensor of type float16, float32, int32, int64, int8, int16, uint8, uint16, uint32, uint64, or bool.

indices: tensor of type int32 or int64.

\[Outputs\]

One output

y: tensor of the identical data type as input x1.

\[Attributes\]

axis: int, the axis in x1 to gather indices from. Must be in the range \[–r, r – 1\], where r indicates the rank of the input x1.

\[Restrictions\]

indices must not be negative.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="gathernd.md">GatherND</h2>

## Description<a name="section12725193815114"></a>

Gathers slices of data into an output tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

data: input tensor of rank r \>= 1. Must be one of the following types: float16, float32, double, int32, int64.

indices: tensor of type int64, of rank q \>= 1.

\[Outputs\]

One output

output: tensor of rank q + r – indices\_shape\[–1\] – 1

\[Attributes\]

batch\_dims: int, the number of batch dimensions. Defaults to 0.

\[Restrictions\]

The operator does not support inputs of type double when the atc command-line option --precision\_mode is set to must\_keep\_origin\_dtype.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v11/v12/v13

<h2 id="gatherelements.md">GatherElements</h2>

## Description<a name="section12725193815114"></a>

Produces an output by indexing into the input tensor at index positions.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

input: input tensor of rank \> 1. Must be one of the following types: float16, float32, double, int32, int64.

indices: tensor of type int32 or int64.

\[Outputs\]

One output

output: tensor with the same shape as indices.

\[Attributes\]

axis: int, the axis to gather on. Defaults to 0.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="gemm.md">Gemm</h2>

## Description<a name="section12725193815114"></a>

General matrix multiplication

## Parameters<a name="section9981612134"></a>

\[Inputs\]

A: 2D tensor of type float16 or float32.

B: 2D tensor of type float16 or float32.

C: \(optional\) bias, not supported currently.

\[Outputs\]

Y: 2D tensor of type float16 or float32.

\[Attributes\]

transA: bool, indicating whether A needs to be transposed.

transB: bool, indicating whether B needs to be transposed.

alpha: float, not supported currently.

beta: float, not supported currently.

\[Restrictions\]

Opset V8, V9, and V10 versions do not support inputs of type float32 when the atc command-line option  **--precision\_mode**  is set to  **must\_keep\_origin\_dtype**.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="globalaveragepool.md">GlobalAveragePool</h2>

## Description<a name="section12725193815114"></a>

Performs global average pooling.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

X: tensor of type float16 or float32, in NCHW format.

\[Outputs\]

Y: pooled tensor in NCHW format. Has the same data type as X.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="globallppool.md">GlobalLpPool</h2>

## Description<a name="section12725193815114"></a>

Performs global norm pooling.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

input: tensor of type float16 or float32.

\(Optional\) p: int32. Defaults to  **2**.

\[Outputs\]

One output

y: tensor of the same data type as input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="globalmaxpool.md">GlobalMaxPool</h2>

## Description<a name="section12725193815114"></a>

Performs global max pooling.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: output tensor of the upstream node. Must be of type float16, float32, or double.

\[Outputs\]

One output

output: pooled tensor

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="greater.md">Greater</h2>

## Description<a name="section12725193815114"></a>

Returns the truth value of \(x1 \> x2\) element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x1: tensor of type float16, float32, int32, int8, or uint8.

x2: tensor of type float16, float32, int32, int8, or uint8.

\[Outputs\]

One output

y: tensor of type bool.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="greaterorequal.md">GreaterOrEqual</h2>

## Description<a name="section12725193815114"></a>

Returns the truth value of \(x1 \>= x2\) element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x1: tensor of type float16, float32, int32, int8, or uint8.

x2: tensor of type float16, float32, int32, int8, or uint8.

\[Outputs\]

One output

y: tensor of type bool.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v12

<h2 id="hardsigmoid.md">HardSigmoid</h2>

## Description<a name="section12725193815114"></a>

Takes one input data \(tensor\) and produces one output data \(tensor\) where the HardSigmoid function, y = max\(0, min\(1, alpha \* x + beta\)\), is applied to the tensor element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

X: tensor of type float16, float, or double.

\[Outputs\]

One output

Y: tensor of type float16, float, or double.

\[Attributes\]

alpha: float. Defaults to 0.2.

beta: float. Defaults to 0.2.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v1/v6/v8/v9/v10/v11/v12/v13

<h2 id="hardmax.md">hardmax</h2>

## Description<a name="section12725193815114"></a>

Computes the hardmax values for the given input: Hardmax\(element in input, axis\) = 1 if the element is the first maximum value along the specified axis, 0 otherwise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32, of rank = 2.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Attributes\]

axis: int. The dimension Hardmax will be performed on. Defaults to –1.

\[Restrictions\]

In the atc command line, the --precision\_mode option must be set to allow\_fp32\_to\_fp16.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="hardswish.md">HardSwish</h2>

## Description<a name="section12725193815114"></a>

Applies the HardSwish function.  **y=x \* max\(0, min\(1, alpha \* x + beta \)\)**, where  **alpha**  is  **1/6**  and  **beat**  is  **0.5**.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of type float16 or float32.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v14

<h2 id="identity.md">Identity</h2>

## Description<a name="section12725193815114"></a>

Identity operator

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="if.md">If</h2>

## Description<a name="section12725193815114"></a>

If conditional

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

cond: condition for the if operator.

Two attributes

else\_branch: branch tensor to run if condition is false.

then\_branch: branch tensor to run if condition is true.

\[Outputs\]

One or more outputs

y: tensor or list of tensors

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="instancenormalization.md">InstanceNormalization</h2>

## Description<a name="section421532641316"></a>

Computes a tensor by using the formula: y = scale \* \(x – mean\) / sqrt\(variance + epsilon\) + B, where mean and variance are computed per instance per channel.

## Parameters<a name="section143631030111310"></a>

\[Inputs\]

Three inputs

x: tensor of type float16 or float.

scale: 1D scale tensor of size C.

B: 1D tensor of size C.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Attributes\]

epsilon: float. The epsilon value to use to avoid division by zero. Defaults to 1e – 05.

## ONNX Opset Support<a name="section19647924181413"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="less.md">Less</h2>

## Description<a name="section12725193815114"></a>

Returns the truth value of \(x1 < x2\) element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x1: tensor of type float16, float32, int32, int8, or uint8.

x2: tensor of type float16, float32, int32, int8, or uint8.

\[Outputs\]

One output

y: tensor of type bool.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="leakyrelu.md">LeakyRelu</h2>

## Description<a name="section12725193815114"></a>

Computes the Leaky ReLU activation function.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

\[Attributes\]

alpha: float, the leakage coefficient. Defaults to 0.01.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="lessorequal.md">LessOrEqual</h2>

## Description<a name="section12725193815114"></a>

Returns the truth value of \(x <= y\) element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x: tensor of type float16 or float32.

y: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of type bool, with the same shape as the input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v12/v13

<h2 id="log.md">Log</h2>

## Description<a name="section12725193815114"></a>

Computes natural logarithm of x element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="logsoftmax.md">LogSoftMax</h2>

## Description<a name="section12725193815114"></a>

Computes log softmax activations.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

\[Attributes\]

axis: int. Must be in the range \[–r, r – 1\], where r indicates the rank of the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="lpnormalization.md">LpNormalization</h2>

## Description<a name="section12725193815114"></a>

Given a matrix, applies Lp-normalization along the provided axis.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

input: tensor of type float16 or float.

\[Outputs\]

One output

output: tensor of type float16 or float.

\[Attributes\]

axis: int. Defaults to  **–1**.

p: int. Defaults to  **2**.

\[Restrictions\]

Beware that both the  **SAME\_UPPER**  and  **SAME\_LOWER**  values of auto\_pad are functionally the same as the SAME argument of built-in TBE operators. The attribute configuration may lead to an accuracy drop as the SAME argument is position-insensitive.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v1/v8/v9/v10/v11/v12/v13

<h2 id="lppool.md">LpPool</h2>

## Description<a name="section12725193815114"></a>

Performs Lp norm pooling.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of type float16 or float32.

\[Attributes\]

auto\_pad: string. Defaults to  **NOTSET**. The value can be  **NOTSET**,  **SAME\_UPPER**, or  **VALID**.

kernel\_shape: int list, size of the kernel on each axis. This parameter is mandatory.

p: int, norm. Defaults to  **2**.

pads: int list.

strides: int list.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v11/v12/v13

<h2 id="lrn.md">LRN</h2>

## Description<a name="section12725193815114"></a>

Performs local response normalization.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type and format as input x.

\[Attributes\]

alpha: float, a scale factor.

beta: float, an exponent.

bias: float.

size: int, the number of channels to sum over. Must be odd.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="lstm.md">LSTM</h2>

## Description<a name="section12725193815114"></a>

Computes a one-layer LSTM. This operator is usually supported via some custom implementation such as CuDNN.

## Parameters<a name="section9981612134"></a>

\[3–8 Inputs\]

X: tensor of type float16, float, or double.

W: tensor of type float16, float, or double.

R: tensor of type float16, float, or double.

B: tensor of type float16, float, or double.

sequence\_lens: tensor of type int32.

initial\_h: tensor of type float16, float, or double.

initial\_c: tensor of type float16, float, or double.

p: tensor of type float16, float, or double.

\[0–3 Outputs\]

Y: tensor of type float16, float, or double.

Y\_h: tensor of type float16, float, or double.

Y\_c: tensor of type float16, float, or double.

\[Attributes\]

activation\_alpha: list of floats.

activation\_beta: list of floats.

activations: list of strings.

clip: float

direction: string. Defaults to forward.

hidden\_size: int

input\_forget: int. Defaults to 0.

layout: int. Defaults to 0.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="matmul.md">MatMul</h2>

## Description<a name="section12725193815114"></a>

Multiplies two matrices.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x1: 2D tensor of type float16.

x2: 2D tensor of type float16.

\[Outputs\]

One output

y: 2D tensor of type float16.

\[Restrictions\]

Only 1D to 6D inputs are supported.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="max.md">Max</h2>

## Description<a name="section12725193815114"></a>

Computes element-wise max of each of the input tensors.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One or more inputs \(1–∞\)

data\_0: list of tensors. Must be one of the following types: float16, float32, int8, int16, int32.

\[Outputs\]

One output

max: tensor with the same type and shape as the input x \(broadcast shape\)

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="maxpool.md">MaxPool</h2>

## Description<a name="section12725193815114"></a>

Performs max pooling.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

X: tensor of type float16 or float32, in NCHW format.

\[Outputs\]

Y: tensor of type float16 or float32, in NCHW format.

\[Attributes\]

auto\_pad: \(optional\) selected from SAME\_UPPER, SAME\_LOWER, VALID, and NOTSET.

storage\_order: not supported currently.

kernel\_shape: \(optional\)

-   kernel\_shape\[0\]: int32, the kernel height. Must be in the range \[1, 32768\]. Defaults to 1.
-   kernel\_shape\[1\]: int32, the kernel width. Must be in the range \[1, 32768\]. Defaults to 1.

strides: \(optional\)

-   strides\[0\]: int32, the stride height. Defaults to 1.
-   strides\[1\]: int32, the stride width. Defaults to 1.

pads: \(optional\)

-   pads\[0\]: int32, top padding. Defaults to 0.
-   pads\[1\]: int32, bottom padding. Defaults to 0.
-   pads\[2\]: int32, left padding. Defaults to 0.
-   pads\[3\]: int32, right padding. Defaults to 0.

ceil\_mode: \(optional\) int32, either 0 \(floor mode\) or 1 \(ceil mode\). Defaults to 0.

\[Restrictions\]

When strides\[0\] or strides\[1\] is greater than 63, computation is performed on AI CPU, which will compromise performance.

When the value of kernel\_shape\_H or kernel\_shape\_W is beyond the range \[1,255\] or kernel\_shape\_H \* kernel\_shape\_W \> 256, computation is performed on AI CPU, which will compromise performance.

input\_w ∈ \[1, 4096\]

When N of the input tensor is a prime number, N < 65535.

dilations is not supported for a 2D tensor.

If auto\_pad is VALID, ceil\_mode must be 0.

The operator does not support inputs of type float32 when the atc command-line option  **--precision\_mode**  is set to  **must\_keep\_origin\_dtype**.

pads and auto\_pad are mutually exclusive.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="maxroipool.md">MaxRoiPool</h2>

## Description<a name="section12725193815114"></a>

Consumes an input tensor X and region of interests \(RoIs\) to apply max pooling across each RoI, to produce output 4-D tensor of shape \(num\_rois, channels, pooled\_shape\[0\], pooled\_shape\[1\]\).

## Parameters<a name="section9981612134"></a>

\[Inputs\]

X: tensor of type float16 or float.

rois: tensor of type float16 or float.

\[Outputs\]

Y: tensor of type float16, float, or double.

\[Attributes\]

pooled\_shape: list of ints

spatial\_scale: float. Defaults to 1.0.

\[Restrictions\]

The operator does not support inputs of type float32 when the atc command-line option  **--precision\_mode**  is set to  **must\_keep\_origin\_dtype**.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/13

<h2 id="maxunpool.md">MaxUnpool</h2>

## Description<a name="section7149182994210"></a>

Indicates the reverse of the MaxPool operation.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

X: tensor of type float16 or float32.

I: tensor of type int64.

\(Optional\) output\_shape: output shape of type int64.

\[Outputs\]

Y: tensor of the same data type as the input.

\[Attributes\]

\(Mandatory\) kernel\_shape: int list, kernel size on each axis.

pads: int list, pad on each axis.

strides: int list, stride on each axis.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v9/v11/v12/v13

<h2 id="mean.md">Mean</h2>

## Description<a name="section12725193815114"></a>

Computes element-wise mean of each of the input tensors \(with NumPy-style broadcasting support\). All inputs and outputs must have the same data type. This operator supports multi-directional \(NumPy-style\) broadcasting.

## Parameters<a name="section9981612134"></a>

\[Inputs\] One or more inputs \(1–∞\)

data\_0: tensor of type float16, float, double, or bfloat16.

\[Outputs\]

mean: tensor of type float16, float, double, or bfloat16.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="meanvariancenormalization.md">MeanVarianceNormalization</h2>

## Description<a name="section12725193815114"></a>

Performs mean variance normalization on the input tensor X using formula: \(X – EX\)/sqrt\(E\(X – EX\)^2\)

## Parameters<a name="section9981612134"></a>

\[Inputs\]

X: tensor of type float16, float, or bfloat16.

\[Outputs\]

Y: tensor of type float16, float, or bfloat16.

\[Attributes\]

axes: list of ints. Defaults to \['0', '2', '3'\].

## ONNX Opset Support<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="min.md">Min</h2>

## Description<a name="section12725193815114"></a>

Returns the minimum of the input tensors.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: list of tensors of type float16 or float32.

\[Outputs\]

One output

y: output tensor

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="mod.md">Mod</h2>

## Description<a name="section12725193815114"></a>

Performs element-wise binary modulus \(with NumPy-style broadcasting support\). The sign of the remainder is the same as that of the divisor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

A: tensor. Must be one of the following data types: uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float, double, bfloat16.

B: tensor. Must be one of the following data types: uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float, double, bfloat16.

\[Outputs\]

C: tensor. Must be one of the following data types: uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float, double, bfloat16.

\[Attributes\]

fmod: int. Defaults to 0.

\[Restrictions\]

fmod must not be 0 if the inputs are of type float.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v10/v11/v12/v13

<h2 id="mul.md">Mul</h2>

## Description<a name="section12725193815114"></a>

Performs dot product of two matrices.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

A: tensor of type float16, float32, uint8, int8, int16, or int32.

B: tensor of type float16, float32, uint8, int8, int16, or int32.

\[Outputs\]

C: tensor of the identical data type as the input tensor.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="multinomial.md">Multinomial</h2>

## Description<a name="section12725193815114"></a>

Generates a tensor of samples from a multinomial distribution according to the probabilities of each of the possible outcomes.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32, with shape \[batch\_size, class\_size\].

\[Outputs\]

One output

y: tensor of type int32 or int64, with shape \[batch\_size, sample\_size\].

\[Attributes\]

dtype: int. The output dtype. Defaults to 6 \(int32\).

sample\_size: int. Number of times to sample. Defaults to 1.

seed: float. Seed to the random generator.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="neg.md">Neg</h2>

## Description<a name="section12725193815114"></a>

Computes numerical negative value element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or int32.

\[Outputs\]

One output

y: tensor of the identical data type as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="nonmaxsuppression.md">NonMaxSuppression</h2>

## Description<a name="section12725193815114"></a>

Filters out boxes that have high intersection-over-union \(IOU\) overlap with previously selected boxes. Bounding boxes with score less than score\_threshold are removed. Bounding box format is indicated by the center\_point\_box attribute. Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system result in the same boxes being selected by the algorithm. The selected\_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes. The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.

## Parameters<a name="section9981612134"></a>

\[2–5 Inputs\]

boxes: tensor of type float

scores: tensor of type float

max\_output\_boxes\_per\_class: \(optional\) tensor of type int64

iou\_threshold: \(optional\) tensor of type float

score\_threshold: \(optional\) tensor of type float

\[Outputs\]

selected\_indices: tensor of type int64

\[Attributes\]

center\_point\_box: int. Defaults to 0.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v10/v11/v12/v13

<h2 id="nonzero.md">NonZero</h2>

## Description<a name="section12725193815114"></a>

Returns the indices of the elements that are non-zero \(in row-major order\).

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, int32, int8, or uint8.

\[Outputs\]

One output

y: tensor of type int64.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="not.md">Not</h2>

## Description<a name="section12725193815114"></a>

Returns the negation of the input tensor element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type bool.

\[Outputs\]

One output

y: tensor of type bool.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="onehot.md">OneHot</h2>

## Description<a name="section12725193815114"></a>

Produces a one-hot tensor based on inputs.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Three inputs

indices: tensor. Must be one of the following data types: uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float, double.

depth: tensor. Must be one of the following data types: uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float, double.

values: tensor. Must be one of the following data types: uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float, double.

\[Attributes\]

One attribute

axis: \(optional\) axis along which one-hot representation is added.

\[Outputs\]

One output

y: tensor of the identical data type as the values input.

\[Restrictions\]

axis must not be less than –1.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="or.md">Or</h2>

## Description<a name="section12725193815114"></a>

Returns the tensor resulted from performing the or logical operation element-wise on the input tensors.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

X1: tensor of type bool.

X2: tensor of type bool.

\[Outputs\]

One output

y: tensor of type bool.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="randomnormallike.md">RandomNormalLike</h2>

## Description<a name="section421532641316"></a>

Generates a tensor with random values drawn from a normal distribution. The shape of the output tensor is copied from the shape of the input tensor.

## Parameters<a name="section143631030111310"></a>

\[Inputs\]

One input

x: tensor of type float16 or float.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Attributes\]

dtype: int, specifying the data type of the output tensor.

mean: float. The mean of the normal distribution. Defaults to 0.0.

scale: float. The standard deviation of the normal distribution. Defaults to 1.0.

seed: float. Seed to the random generator.

## ONNX Opset Support<a name="section19647924181413"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="randomuniformlike.md">RandomUniformLike</h2>

## Description<a name="section421532641316"></a>

Generates a tensor with random values drawn from a uniform distribution. The shape of the output tensor is copied from the shape of the input tensor.

## Parameters<a name="section143631030111310"></a>

\[Inputs\]

One input

x: tensor of type float16 or float.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Attributes\]

dtype: int, specifying the data type of the output tensor.

high: float. Upper boundary of the uniform distribution. Defaults to 1.0.

low: float. Lower boundary of the uniform distribution. Defaults to 0.0.

seed: float. Seed to the random generator.

## ONNX Opset Support<a name="section19647924181413"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="randomuniform.md">RandomUniform</h2>

## Description<a name="section12725193815114"></a>

Generates a tensor with random values drawn from a uniform distribution.

## Parameters<a name="section9981612134"></a>

\[Attributes\]

Five attributes

dtype: int. Specifies the output data type.

high: float. Specifies the upper boundary.

low: float. Specifies the lower boundary.

seed: \(optional\) seed to the random generator.

shape: output shape.

\[Outputs\]

One output

y: tensor of the data type specified by the dtype attribute.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="range.md">Range</h2>

## Description<a name="section12725193815114"></a>

Generate a tensor containing a sequence of numbers.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Three inputs

start: scalar of type float16 or float32.

limit: scalar of type float16 or float32.

delta: scalar of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type as input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reciprocal.md">Reciprocal</h2>

## Description<a name="section12725193815114"></a>

Computes the reciprocal of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reducel1.md">ReduceL1</h2>

## Description<a name="section12725193815114"></a>

Computes the L1 norm of the input tensor's elements along the provided axes. The resulted tensor has the same rank as the input if keepdim is set to 1. If keepdim is set to 0, then the result tensor has the reduced dimension pruned. The above behavior is similar to NumPy, with the exception that NumPy defaults keepdim to False instead of True.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

data: tensor. Must be one of the following types: uint32, uint64, int32, int64, float16, float, double, bfloat16.

\[Outputs\]

reduced: tensor. Must be one of the following types: uint32, uint64, int32, int64, float16, float, double, bfloat16.

\[Attributes\]

axes: list of ints.

keepdims: int. Defaults to 1.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reducel2.md">ReduceL2</h2>

## Description<a name="section12725193815114"></a>

Computes the L2 norm of the input tensor's elements along the provided axes. The resulted tensor has the same rank as the input if keepdim is set to 1. If keepdim is set to 0, then the result tensor has the reduced dimension pruned. The above behavior is similar to NumPy, with the exception that NumPy defaults keepdim to False instead of True.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

data: tensor. Must be one of the following types: uint32, uint64, int32, int64, float16, float, double, bfloat16.

\[Outputs\]

reduced: tensor. Must be one of the following types: uint32, uint64, int32, int64, float16, float, double, bfloat16.

\[Attributes\]

axes: list of ints.

keepdims: int. Defaults to 1.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reducelogsum.md">ReduceLogSum</h2>

## Description<a name="section12725193815114"></a>

Computes the sum of elements across dimensions of a tensor in log representations.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of type float16 or float32.

\[Attributes\]

axes: int list. Must be in the range \[–r, r – 1\], where  **r**  indicates the dimension count of the input x.

keepdims: int. Defaults to  **1**, meaning that the reduced dimensions with length 1 are retained.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v11/v13

<h2 id="reducelogsumexp.md">ReduceLogSumExp</h2>

## Description<a name="section12725193815114"></a>

Reduces a dimension of a tensor by calculating exponential for all elements in the dimension and calculates logarithm of the sum.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

data: tensor of type float16 or float32.

\[Outputs\]

One output

reduced: tensor of type float16 or float32.

\[Attributes\]

axes: tensor of type int32 or int64. Must be in the range \[–r, r – 1\], where  **r**  indicates the dimension count of the input x.

keepdims: int, indicating whether to reduce the dimensions. The default value is  **1**, indicating that the dimensions are reduced.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reducemin.md">ReduceMin</h2>

## Description<a name="section12725193815114"></a>

Computes the minimum of elements across dimensions of a tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of type float16 or float32.

\[Attributes\]

axes: int list. Must be in the range \[–r, r – 1\], where  **r**  indicates the dimension count of the input x.

keepdims: int. Defaults to 1, meaning that the reduced dimensions with length 1 are retained.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reducemean.md">ReduceMean</h2>

## Description<a name="section12725193815114"></a>

Computes the mean of elements across dimensions of a tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type and format as input x.

\[Attributes\]

axes: 1D list of ints, the dimensions to reduce. Must be in the range \[–r, r – 1\], where r indicates the rank of the input.

keepdims: int. Defaults to 1, meaning that the reduced dimensions with length 1 are retained.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reduceprod.md">ReduceProd</h2>

## Description<a name="section12725193815114"></a>

Computes the product of the input tensor's elements along the provided axes. The resulted tensor has the same rank as the input if keepdim is set to 1. If keepdim is set to 0, then the result tensor has the reduced dimension pruned.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

data: tensor. Must be one of the following types: uint32, uint64, int32, int64, float16, float, double, bfloat16.

\[Outputs\]

reduced: tensor. Must be one of the following types: uint32, uint64, int32, int64, float16, float, double, bfloat16.

\[Attributes\]

axes: list of ints.

keepdims: int. Defaults to 1.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reducesumsquare.md">ReduceSumSquare</h2>

## Description<a name="section12725193815114"></a>

Computes the sum square of the input tensor's elements along the provided axes. The resulted tensor has the same rank as the input if keepdim is set to 1. If keepdim is set to 0, then the result tensor has the reduced dimension pruned. The above behavior is similar to NumPy, with the exception that NumPy defaults keepdim to False instead of True.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

data: tensor. Must be one of the following types: uint32, uint64, int32, int64, float16, float, double, bfloat16.

\[Outputs\]

reduced: tensor. Must be one of the following types: uint32, uint64, int32, int64, float16, float, double, bfloat16.

\[Attributes\]

axes: list of ints.

keepdims: int. Defaults to 1.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v1/v8/v9/v10/v11/v12/v13

<h2 id="resize.md">Resize</h2>

## Description<a name="section12725193815114"></a>

Resizes the input tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Four inputs

x: tensor of type float16 or float32.

roi: 1D tensor of type float16 or float32, with shape \[start1, ..., startN, end1, ..., endN\]. The tensor normalized by the input image.

scales: array. Has the same rank as that of the input x.

sizes: size of the output tensor.

\[Outputs\]

One output

y: resized tensor

\[Attributes\]

coordinate\_transformation\_mode: string. Defaults to half\_pixel. Describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor.

cubic\_coeff\_a: float. The coefficient used in cubic interpolation. Defaults to –0.75.

exclude\_outside: int. The weight outside the tensor. Defaults to 0.

mode: string. Interpolation mode selected from nearest \(default\), linear, and cubic.

nearest\_mode: string. Defaults to round\_prefer\_floor.

\[Restrictions\]

Currently, only the nearest and linear interpolation modes are supported to process images. In addition, the model's two inputs \(scales and sizes\) need to be changed from placeholders to constants. You can use ONNX Simplifier to simplify your model.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v10/v11/v12

<h2 id="relu.md">Relu</h2>

## Description<a name="section12725193815114"></a>

Applies the rectified linear unit activation function.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

X: input tensor of type float32, int32, uint8, int16, int8, uint16, float16, or qint8.

\[Outputs\]

Y: tensor of the identical data type as X.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reducesum.md">ReduceSum</h2>

## Description<a name="section12725193815114"></a>

Computes the sum of the input tensor's element along the provided axes.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type and format as input x.

\[Attributes\]

axes: 1D list of ints, the dimensions to reduce. Must be in the range \[–r, r – 1\], where r indicates the rank of the input.

keepdims: int. Defaults to 1, meaning that the reduced dimensions with length 1 are retained.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reducemax.md">ReduceMax</h2>

## Description<a name="section12725193815114"></a>

Computes the maximum of elements across dimensions of a tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or int32.

\[Outputs\]

One output

y: tensor of type float16, float32, or int32.

\[Attributes\]

axes: list of ints. Must be in the range \[–r, r – 1\], where r indicates the rank of the input.

keepdims: int. Defaults to 1, meaning that the reduced dimensions with length 1 are retained.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reshape.md">Reshape</h2>

## Description<a name="section12725193815114"></a>

Reshapes the input.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

data: tensor.

shape: tensor of type int64, for the shape of the output tensor.

\[Outputs\]

reshaped: tensor

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="reversesequence.md">ReverseSequence</h2>

## Description<a name="section12725193815114"></a>

Reverses batch of sequences having different lengths.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x: tensor of type float16 or float32, of rank \>= 2.

sequence\_lens: tensor of type int64. Lengths of the sequences in a batch.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Attributes\]

batch\_axis: int. Specifies the batch axis. Defaults to 1.

time\_axis: int. Specifies the time axis. Defaults to 1.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v10/v11/v12/v13

<h2 id="roiextractor.md">RoiExtractor</h2>

## Description<a name="section12725193815114"></a>

Obtains the ROI feature matrix from the feature mapping list.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

features: tensor of type float32 or float16.

rois: tensor of type float32 or float16.

\[Attributes\]

Eight attributes

finest\_scale: int

roi\_scale\_factor: float

spatial\_scale: array of floats

pooled\_height: int

pooled\_width: int

sample\_num: int

pool\_mode: string

aligned: bool

\[Outputs\]

One output

y: tensor of type float32 or float16.

## ONNX Opset Support<a name="section13311501226"></a>

No ONNX support for this custom operator

<h2 id="roialign.md">RoiAlign</h2>

## Description<a name="section12725193815114"></a>

Performs ROI align operation.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Three inputs

x: 4D tensor of type float16 or float32.

rois: float16 or float32. ROIs to pool over. Has shape \(num\_rois, 4\).

batch\_indices: int64. Has shape \(num\_rois,\).

\[Outputs\]

One output

y: tensor of the identical type as input x. Has shape \(num\_rois, C, output\_height, output\_width\).

\[Attributes\]

mode: string. The pooling method. Defaults to avg.

output\_height: int. Pooled output y's height. Defaults to 1.

output\_width: int. Pooled output y's width. Defaults to 1.

sampling\_ratio: int. Number of sampling points in the interpolation grid used to compute the output value of each pooled output bin. Defaults to 0.

spatial\_scale: float. Multiplicative spatial scale factor to translate ROI coordinates from their input spatial scale to the scale used when pooling. Defaults to 1.0.

\[Restrictions\]

batch\_indices must be of type int32 instead of int64.

The operator does not support inputs of type float32 or float64 when the atc command-line option  **--precision\_mode**  is set to  **must\_keep\_origin\_dtype**.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v10/v11/v12/v13

<h2 id="round.md">Round</h2>

## Description<a name="section12725193815114"></a>

Rounds the values of a tensor to the nearest integer, element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="prelu.md">PRelu</h2>

## Description<a name="section12725193815114"></a>

Computes Parametric Rectified Linear Unit.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x: tensor of type float16 or float32.

slope: tensor of the same data type as input x.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Restrictions\]

slope must be 1D. When input x is 1D, the dimension value of slope must be 1. When input x is not 1D, the dimension value of slope can be 1 or shape\[1\] of input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="scatter.md">Scatter</h2>

## Description<a name="section421532641316"></a>

Returns the result by updating the values of the input data to values specified by updates at specific index positions specified by indices.

## Parameters<a name="section143631030111310"></a>

\[Inputs\]

Three inputs

data: tensor of type float16, float, or int32.

indices: tensor of type int32 or int64.

updates: tensor of the identical data type as data.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Attributes\]

axis: int, specifying which axis to scatter on. Defaults to 0.

## ONNX Opset Support<a name="section19647924181413"></a>

Opset v9/v10

<h2 id="scatterelements.md">ScatterElements</h2>

## Description<a name="section421532641316"></a>

Returns the result by updating the values of the input data to values specified by updates at specific index positions specified by indices.

## Parameters<a name="section143631030111310"></a>

\[Inputs\]

One input

data: tensor of type float16, float, or int32.

indices: tensor of type int32 or int64.

updates: tensor of the identical data type as data.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Attributes\]

axis: int, specifying which axis to scatter on. Defaults to 0.

## ONNX Opset Support<a name="section19647924181413"></a>

Opset v11/v12/v13

<h2 id="scatternd.md">ScatterND</h2>

## Description<a name="section12725193815114"></a>

Creates a copy of the input data, and then updates its values to those specified by updates at specific index positions specified by indices.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Three inputs

data: tensor of type float16 or float32, of rank \>= 1.

indices: tensor of type int64, of rank \>= 1.

updates: tensor of type float16 or float32, of rank = q + r – indices\_shape\[–1\] – 1.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v11

<h2 id="shrink.md">Shrink</h2>

## Description<a name="section421532641316"></a>

Takes one input tensor and outputs one tensor. The formula of this operator is: If x < – lambd, y = x + bias; If x \> lambd, y = x – bias; otherwise, y = 0.

## Parameters<a name="section143631030111310"></a>

\[Inputs\]

One input

data: tensor of type float16 or float.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Attributes\]

bias: float. Defaults to 0.0.

lambda: float. Defaults to 0.5.

## ONNX Opset Support<a name="section19647924181413"></a>

Opset v9/v10/v11/ v12/v13

<h2 id="selu.md">Selu</h2>

## Description<a name="section12725193815114"></a>

Produces a tensor where the scaled exponential linear unit function: y = gamma \* \(alpha \* e^x – alpha\) for x <= 0, y = gamma \* x for x \> 0, is applied to the input tensor element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

Two attributes

alpha: coefficient of SELU

gamma: coefficient of SELU

\[Outputs\]

One output

y: tensor of the identical data type as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="shape.md">Shape</h2>

## Description<a name="section12725193815114"></a>

Returns a tensor containing the shape of the input tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor

\[Outputs\]

y: int64 tensor containing the shape of the input tensor.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="sigmoid.md">Sigmoid</h2>

## Description<a name="section12725193815114"></a>

Computes sigmoid of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type as input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="slice.md">Slice</h2>

## Description<a name="section12725193815114"></a>

Extracts a slice from a tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Five inputs

x: tensor of type float16, float32, int32, uint8, bool, or int8.

starts: 1D tensor of type int32 or int64, specifying the start index.

ends: 1D tensor of type int32 or int64, specifying the end index.

axes: \(optional\) 1D tensor of type int32 or int64. The axis to extract a slice from. Must be in the range \[–r, r – 1\], where r indicates the rank of the input x.

steps: \(optional\) 1D tensor of type int32 or int64, specifying the slice step. The slice step of the last axis must be 1.

\[Outputs\]

y: tensor of the identical data type as input x.

\[Restrictions\]

x: must have a rank greater than 1.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="softmax.md">Softmax</h2>

## Description<a name="section12725193815114"></a>

Computes softmax activations.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input x.

\[Attributes\]

axis: \(optional\) int, the dimension softmax would be performed on. Defaults to –1. Must be in the range \[–len\(x.shape\), len\(x.shape\) – 1\].

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="softsign.md">Softsign</h2>

## Description<a name="section12725193815114"></a>

Computes softsign: \(x/\(1+|x|\)\)

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="softplus.md">Softplus</h2>

## Description<a name="section12725193815114"></a>

Computes softplus.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

X: 1D input tensor

\[Outputs\]

One output

Y: 1D tensor

\[Restrictions\]

Only the float16 and float32 data types are supported.

The output has the identical data type as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="spacetodepth.md">SpaceToDepth</h2>

## Description<a name="section12725193815114"></a>

Rearranges blocks of spatial data into depth. More specifically, this operator outputs a copy of the input tensor where values from the height and width dimensions are moved to the depth dimension.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

input: tensor. Must be one of the following data types: uint8, uint16, uint32, uint64, int8, int16, int32, int64, bfloat16, float16, float, double, string, bool, complex64, complex128.

\[Outputs\]

output: tensor. Must be one of the following data types: uint8, uint16, uint32, uint64, int8, int16, int32, int64, bfloat16, float16, float, double, string, bool, complex64, complex128.

\[Attributes\]

blocksize: int

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="split.md">Split</h2>

## Description<a name="section12725193815114"></a>

Splits the input tensor into a list of sub-tensors.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.

\[Outputs\]

One output

y: list of tensors of the identical data type as input x.

\[Attributes\]

split: list of type int8, int16, int32, or int64, for the length of each output along axis.

axis: int8, int16, int32, or int64, for the axis along which to split.

\[Restrictions\]

Each element of split must be greater than or equal to 1.

The sum of all split elements must be equal to axis.

axis ∈ \[–len\(x.shape\), len\(x.shape\) – 1\]

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="sqrt.md">Sqrt</h2>

## Description<a name="section12725193815114"></a>

Computes element-wise square root of the input tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor

\[Outputs\]

One output

y: tensor

\[Restrictions\]

The output has the identical shape and dtype as the input. The supported data types are float16 and float32.

NaN is returned if x is less than 0.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="squeeze.md">Squeeze</h2>

## Description<a name="section12725193815114"></a>

Removes dimensions of size 1 from the shape of a tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor. Must be one of the following data types: float16, float32, double, uint8, uint16, uint32, uint64, int8, int16, int32, int64, bool.

\[Outputs\]

y: tensor of the identical data type as the input.

\[Attributes\]

axes: 1D list of int32s or int64s, indicating the dimensions to squeeze. Negative value means counting dimensions from the back. Accepted range is \[–r, r – 1\] where r = rank\(x\).

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="sub.md">Sub</h2>

## Description<a name="section12725193815114"></a>

Performs element-wise subtraction.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x1: tensor

x2: tensor

\[Outputs\]

One output

y: tensor of the identical data type as the input.

\[Restrictions\]

The output has the identical shape and dtype as the input. The supported data types are int32, float16, and float32.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="sign.md">Sign</h2>

## Description<a name="section12725193815114"></a>

Computes the symbol of the input tensor element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="sin.md">Sin</h2>

## Description<a name="section12725193815114"></a>

Computes sine of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="sinh.md">Sinh</h2>

## Description<a name="section12725193815114"></a>

Computes hyperbolic sine of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16, float32, or double.

\[Outputs\]

One output

y: tensor. Has the identical data type and shape as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="size.md">Size</h2>

## Description<a name="section12725193815114"></a>

Outputs the number of elements in the input tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: scalar of type int64

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="sum.md">Sum</h2>

## Description<a name="section12725193815114"></a>

Computes element-wise sum of each of the input tensors.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="tanh.md">Tanh</h2>

## Description<a name="section12725193815114"></a>

Computes hyperbolic tangent of the input element-wise.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type as the input.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="tfidfvectorizer.md">TfIdfVectorizer</h2>

## Description<a name="section421532641316"></a>

Extracts n-grams from the input sequence and save them as a vector.

## Parameters<a name="section143631030111310"></a>

\[Inputs\]

One input

data: tensor of type int32 or int64.

\[Outputs\]

One output

y: tensor of type float.

\[Attributes\]

max\_gram\_length: int. Maximum n-gram length.

max\_skip\_count: int. Maximum number of items to be skipped when constructing an n-gram from data.

min\_gram\_length: int. Minimum n-gram length.

mode: string. The weighting criteria. It can be "TF" \(term frequency\), "IDF" \(inverse document frequency\), or "TFIDF" \(the combination of TF and IDF\).

ngram\_counts: list of ints. The starting indexes of n-grams in pool. It is useful when determining the boundary between two consecutive collections of n-grams.

ngram\_indexes: list of ints. The i-th element in ngram\_indexes indicates the coordinate of the i-th n-gram in the output tensor.

pool\_int64s: list of ints, indicating n-grams learned from the training set. This attribute and pool\_strings are mutually exclusive.

pool\_strings: list of strings. Has the same meaning as pool\_int64s.

weights: list of floats. Stores the weight of each n-gram in pool.

## ONNX Opset Support<a name="section19647924181413"></a>

Opset v9/v10/v11/ v12/v13

<h2 id="tile.md">Tile</h2>

## Description<a name="section12725193815114"></a>

Constructs a tensor by tiling a given tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x: tensor

repeats: 1D tensor of type int64. Has the same size as the number of dimensions in x.

\[Outputs\]

One output

y: tensor of the identical type and dimension as the input. output\_dim\[i\] = input\_dim\[i\] \* repeats\[i\]

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="thresholdedrelu.md">ThresholdedRelu</h2>

## Description<a name="section12725193815114"></a>

When x \> alpha, y = x; otherwise, y = 0.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type float16 or float32.

\[Outputs\]

One output

y: tensor of the identical data type and shape as input x.

\[Attributes\]

alpha: float, indicating the threshold. Defaults to 1.0.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v10/v11/v12/v13

<h2 id="topk.md">TopK</h2>

## Description<a name="section12725193815114"></a>

Retrieves the top-K largest or smallest elements along a specified axis.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x: tensor of type float16 or float32.

k: tensor of type int64.

\[Outputs\]

Two outputs

Values: tensor containing top K values from the input tensor.

Indexes: tensor containing the corresponding input tensor indices for the top K values.

\[Attributes\]

axis: int. The dimension on which to do the sort. Defaults to –1.

largest: int. Whether to return the top-K largest or smallest elements. Defaults to 1.

sorted: int. Whether to return the elements in sorted order. Defaults to 1.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="transpose.md">Transpose</h2>

## Description<a name="section12725193815114"></a>

Transposes the input.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

data: tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64.

\[Outputs\]

transposed: tensor after transposition.

\[Attributes\]

perm: \(required\) list of integers, for the dimension sequence of data.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="pad.md">Pad</h2>

## Description<a name="section12725193815114"></a>

Pads a tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x: tensor of type float16, float32, or int32.

pads: tensor of type int32 or int64.

constant\_value: optional. Defaults to  **0**, an empty string, or  **False**. If the selected mode is  **constant**, the scalar value is used.

\[Outputs\]

One output

y: tensor of the identical data type as input x.

\[Attributes\]

mode: str type. The following modes are supported: constant, reflect, and edge.

\[Restrictions\]

If the value of mode is  **constant**, the value of  **constant\_value**  can only be  **0**.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v11

<h2 id="pow.md">Pow</h2>

## Description<a name="section12725193815114"></a>

Computes x1 to the x2th power.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

x1: tensor of type float16, float32, double, int32, int8, or uint8.

x2: tensor of the identical data type as input x1.

\[Outputs\]

One output

y: tensor of the identical data type as input x1.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="unsqueeze.md">Unsqueeze</h2>

## Description<a name="section12725193815114"></a>

Inserts single-dimensional entries to the shape of an input tensor.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

One input

x: tensor of type uint8, uint16, uint32, int8, int16, int32, float16, or float32.

\[Outputs\]

One output

y: tensor of the identical data type as input x.

\[Attributes\]

axes: list of integers indicating the dimensions to be inserted. Accepted range is \[–input\_rank, input\_rank\]\(inclusive\) where r = rank\(x\).

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/10/v11/v12

<h2 id="xor.md">Xor</h2>

## Description<a name="section12725193815114"></a>

Computes the element-wise logical XOR of the given input tensors.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Two inputs

a: tensor of type bool.

b: tensor of type bool.

\[Outputs\]

c: tensor of type bool.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="where.md">Where</h2>

## Description<a name="section12725193815114"></a>

Returns elements chosen from x or y depending on condition.

## Parameters<a name="section9981612134"></a>

\[Inputs\]

Three inputs

condition: bool.

x: tensor of type float16, float32, int8, int32, or uint8. Elements from which to choose when condition is true.

y: tensor of the identical data type as x. Elements from which to choose when condition is false.

\[Outputs\]

Tensor of the identical data type as input x.

## ONNX Opset Support<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

