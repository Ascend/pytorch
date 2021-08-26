# 支持ONNX算子清单
-   [Abs](#Abs.md)
-   [Acos](#Acos.md)
-   [Acosh](#Acosh.md)
-   [AdaptiveAvgPool2D](#AdaptiveAvgPool2D.md)
-   [AdaptiveMaxPool2D](#AdaptiveMaxPool2D.md)
-   [Add](#Add.md)
-   [Addcmul](#Addcmul.md)
-   [AffineGrid](#AffineGrid.md)
-   [And](#And.md)
-   [Argmax](#Argmax.md)
-   [Argmin](#Argmin.md)
-   [AscendRequantS16](#AscendRequantS16.md)
-   [AscendRequant](#AscendRequant.md)
-   [AscendQuant](#AscendQuant.md)
-   [AscendDequantS16](#AscendDequantS16.md)
-   [AscendDequant](#AscendDequant.md)
-   [AscendAntiQuant](#AscendAntiQuant.md)
-   [Asin](#Asin.md)
-   [Asinh](#Asinh.md)
-   [Atan](#Atan.md)
-   [Atanh](#Atanh.md)
-   [AveragePool](#AveragePool.md)
-   [BatchNormalization](#BatchNormalization.md)
-   [BatchMatMul](#BatchMatMul.md)
-   [BatchMultiClassNMS](#BatchMultiClassNMS.md)
-   [BitShift](#BitShift.md)
-   [Cast](#Cast.md)
-   [Ceil](#Ceil.md)
-   [Celu](#Celu.md)
-   [Concat](#Concat.md)
-   [Clip](#Clip.md)
-   [ConvTranspose](#ConvTranspose.md)
-   [Cumsum](#Cumsum.md)
-   [Conv](#Conv.md)
-   [Compress](#Compress.md)
-   [Constant](#Constant.md)
-   [ConstantOfShape](#ConstantOfShape.md)
-   [Cos](#Cos.md)
-   [Cosh](#Cosh.md)
-   [DeformableConv2D](#DeformableConv2D.md)
-   [Det](#Det.md)
-   [DepthToSpace](#DepthToSpace.md)
-   [Div](#Div.md)
-   [Dropout](#Dropout.md)
-   [Elu](#Elu.md)
-   [EmbeddingBag](#EmbeddingBag.md)
-   [Equal](#Equal.md)
-   [Erf](#Erf.md)
-   [Exp](#Exp.md)
-   [Expand](#Expand.md)
-   [EyeLike](#EyeLike.md)
-   [Flatten](#Flatten.md)
-   [Floor](#Floor.md)
-   [Gather](#Gather.md)
-   [GatherND](#GatherND.md)
-   [GatherElements](#GatherElements.md)
-   [Gemm](#Gemm.md)
-   [GlobalAveragePool](#GlobalAveragePool.md)
-   [GlobalLpPool](#GlobalLpPool.md)
-   [GlobalMaxPool](#GlobalMaxPool.md)
-   [Greater](#Greater.md)
-   [GreaterOrEqual](#GreaterOrEqual.md)
-   [HardSigmoid](#HardSigmoid.md)
-   [hardmax](#hardmax.md)
-   [HardSwish](#HardSwish.md)
-   [Identity](#Identity.md)
-   [If](#If.md)
-   [InstanceNormalization](#InstanceNormalization.md)
-   [Less](#Less.md)
-   [LeakyRelu](#LeakyRelu.md)
-   [LessOrEqual](#LessOrEqual.md)
-   [Log](#Log.md)
-   [LogSoftMax](#LogSoftMax.md)
-   [LpNormalization](#LpNormalization.md)
-   [LpPool](#LpPool.md)
-   [LRN](#LRN.md)
-   [LSTM](#LSTM.md)
-   [MatMul](#MatMul.md)
-   [Max](#Max.md)
-   [MaxPool](#MaxPool.md)
-   [MaxRoiPool](#MaxRoiPool.md)
-   [Mean](#Mean.md)
-   [MeanVarianceNormalization](#MeanVarianceNormalization.md)
-   [Min](#Min.md)
-   [Mod](#Mod.md)
-   [Mul](#Mul.md)
-   [Multinomial](#Multinomial.md)
-   [Neg](#Neg.md)
-   [NonMaxSuppression](#NonMaxSuppression.md)
-   [NonZero](#NonZero.md)
-   [Not](#Not.md)
-   [OneHot](#OneHot.md)
-   [Or](#Or.md)
-   [RandomNormalLike](#RandomNormalLike.md)
-   [RandomUniformLike](#RandomUniformLike.md)
-   [RandomUniform](#RandomUniform.md)
-   [Range](#Range.md)
-   [Reciprocal](#Reciprocal.md)
-   [ReduceL1](#ReduceL1.md)
-   [ReduceL2](#ReduceL2.md)
-   [ReduceLogSum](#ReduceLogSum.md)
-   [ReduceLogSumExp](#ReduceLogSumExp.md)
-   [ReduceMin](#ReduceMin.md)
-   [ReduceMean](#ReduceMean.md)
-   [ReduceProd](#ReduceProd.md)
-   [ReduceSumSquare](#ReduceSumSquare.md)
-   [Resize](#Resize.md)
-   [Relu](#Relu.md)
-   [ReduceSum](#ReduceSum.md)
-   [ReduceMax](#ReduceMax.md)
-   [Reshape](#Reshape.md)
-   [ReverseSequence](#ReverseSequence.md)
-   [RoiExtractor](#RoiExtractor.md)
-   [RoiAlign](#RoiAlign.md)
-   [Round](#Round.md)
-   [PRelu](#PRelu.md)
-   [Scatter](#Scatter.md)
-   [ScatterElements](#ScatterElements.md)
-   [ScatterND](#ScatterND.md)
-   [Shrink](#Shrink.md)
-   [Selu](#Selu.md)
-   [Shape](#Shape.md)
-   [Sigmoid](#Sigmoid.md)
-   [Slice](#Slice.md)
-   [Softmax](#Softmax.md)
-   [Softsign](#Softsign.md)
-   [Softplus](#Softplus.md)
-   [SpaceToDepth](#SpaceToDepth.md)
-   [Split](#Split.md)
-   [Sqrt](#Sqrt.md)
-   [Squeeze](#Squeeze.md)
-   [Sub](#Sub.md)
-   [Sign](#Sign.md)
-   [Sin](#Sin.md)
-   [Sinh](#Sinh.md)
-   [Size](#Size.md)
-   [Sum](#Sum.md)
-   [Tanh](#Tanh.md)
-   [TfIdfVectorizer](#TfIdfVectorizer.md)
-   [Tile](#Tile.md)
-   [ThresholdedRelu](#ThresholdedRelu.md)
-   [TopK](#TopK.md)
-   [Transpose](#Transpose.md)
-   [Pad](#Pad.md)
-   [Pow](#Pow.md)
-   [Unsqueeze](#Unsqueeze.md)
-   [Xor](#Xor.md)
-   [Where](#Where.md)
<h2 id="Abs.md">Abs</h2>

## 功能<a name="section12725193815114"></a>

对输入张量取绝对值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double、int32、int64

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Acos.md">Acos</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的反余弦值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Acosh.md">Acosh</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的反双曲余弦值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="AdaptiveAvgPool2D.md">AdaptiveAvgPool2D</h2>

## 功能<a name="section12725193815114"></a>

对输入进行2d自适应平均池化计算

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【属性】

一个属性：

output\_size：int型数组，指定输出的hw的shape大小

【输出】

一个输出

y：一个tensor，数据类型：与x类型一致

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="AdaptiveMaxPool2D.md">AdaptiveMaxPool2D</h2>

## 功能<a name="section12725193815114"></a>

对输入进行2d自适应最大池化计算

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、float64

【属性】

一个属性：

output\_size：int型数组，指定输出的hw的shape大小

【输出】

两个输出

y：一个tensor，数据类型：与x类型一致

argmax：一个tensor，数据类型：int32，int64

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="Add.md">Add</h2>

## 功能<a name="section12725193815114"></a>

按元素求和

## 边界<a name="section9981612134"></a>

【输入】

两个输入

A：一个张量，数据类型：int8、int16、int32、int64、uint8、float32、float16、double

B：一个张量，数据类型与A相同

【输出】

C：一个张量，数据类型与A相同

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Addcmul.md">Addcmul</h2>

## 功能<a name="section12725193815114"></a>

元素级计算\(x1 \* x2\) \* value + input\_data

## 边界<a name="section9981612134"></a>

【输入】

四个输入

input\_data：一个tensor，数据类型：float16、float32、int32、int8、uint8

x1： 一个tensor，类型与input\_data相同

x2： 一个tensor，类型与input\_data相同

value： 一个tensor，类型与input\_data相同

【输出】

一个输出

y：一个tensor，数据类型：y与输入相同

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="AffineGrid.md">AffineGrid</h2>

## 功能<a name="section12725193815114"></a>

给定一批矩阵，生成采样网格

## 边界<a name="section9981612134"></a>

【输入】

俩个输入

theta：一个tensor，数据类型：float16、float32

output\_size：一个tensor，数据类型：int32

【属性】

一个属性：

align\_corners：bool型

【输出】

一个输出

y：一个tensor，数据类型：int

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="And.md">And</h2>

## 功能<a name="section12725193815114"></a>

逻辑与

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：bool

x2：一个tensor，数据类型：bool

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Argmax.md">Argmax</h2>

## 功能<a name="section12725193815114"></a>

返回指定轴上最大值所对应的索引

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，表示最大值的索引位置，维度比输入x少1，数据类型：int32

【属性】

axis：必选，表示计算最大值索引的方向，数据类型：int32，aixs的值为\[-len\(x.shape\), len\(x.shape\)-1\]

keep\_dim：可选，keep\_dim默认为1，支持1或0。

【约束】

算子不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时fp32类型输入

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Argmin.md">Argmin</h2>

## 功能<a name="section12725193815114"></a>

返回输入张量指定轴上最小值对应的索引

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型：int64

【属性】

axis：数据类型为int，含义：指定计算轴；取值范围：\[-r, r-1\]，r表示输入数据的秩

【约束】

算子不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时fp32类型输入

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="AscendRequantS16.md">AscendRequantS16</h2>

## 功能<a name="section12725193815114"></a>

重新量化算子

## 边界<a name="section9981612134"></a>

【输入】

两个必选输入，一个可选输入

x0：一个tensor，数据类型：int16

req\_scale：一个tensor，数据类型：uint64

x1：一个tensor，数据类型：int16

【属性】

两个属性：

dual\_output：bool型

relu\_flag：bool型

【输出】

两个输出

y0：一个tensor，数据类型：int8

y1：一个tensor，数据类型：int16

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="AscendRequant.md">AscendRequant</h2>

## 功能<a name="section12725193815114"></a>

重新量化算子

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x0：一个tensor，数据类型：int32

req\_scale：一个tensor，数据类型：uint64

【属性】

一个属性：

relu\_flag，数据类型：bool

【输出】

一个输出

y：一个tensor，数据类型：int8

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="AscendQuant.md">AscendQuant</h2>

## 功能<a name="section12725193815114"></a>

量化算子

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16，float32

【属性】

四个属性：

offset，数据类型：float

scale，数据类型：float

sqrt\_mode，数据类型：bool

round\_mode，数据类型：string

【输出】

一个输出

y：一个tensor，数据类型：int8

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="AscendDequantS16.md">AscendDequantS16</h2>

## 功能<a name="section12725193815114"></a>

反量化算子

## 边界<a name="section9981612134"></a>

【输入】

两个必选输入，一个可选输入

x0：一个tensor，数据类型：int32

req\_scale：一个tensor，数据类型：uint64

x1：一个tensor，数据类型：int16

【属性】

一个属性

relu\_flag，数据类型：bool

【输出】

一个输出

y：一个tensor，数据类型：int16

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="AscendDequant.md">AscendDequant</h2>

## 功能<a name="section12725193815114"></a>

反量化算子

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x0：一个tensor，数据类型：int32

deq\_scale：一个tensor，数据类型：uint64,float16

【属性】

sqrt\_mode，数据类型：bool

relu\_flag，数据类型：bool

dtype，数据类型：float

【输出】

一个输出

y：一个tensor，数据类型：float16，float

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="AscendAntiQuant.md">AscendAntiQuant</h2>

## 功能<a name="section12725193815114"></a>

反量化算子

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：int8

【属性】

offset，float型

scale，float型

sqrt\_mode，bool

round\_mode，string

【输出】

一个输出

y：一个tensor，数据类型：float16，float

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="Asin.md">Asin</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的反正弦

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x1：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Asinh.md">Asinh</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量双曲反正弦

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="Atan.md">Atan</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的反正切值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Atanh.md">Atanh</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的双曲反正切

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="AveragePool.md">AveragePool</h2>

## 功能<a name="section12725193815114"></a>

平均池化层

## 边界<a name="section9981612134"></a>

【输入】

X：一个张量，数据类型：float16、float32，格式为NCHW

【输出】

Y：一个张量，数据类型：float16、float32，格式为NCHW

【属性】

auto\_pad：可选，支持NOTSET、SAME\_UPPER、SAME\_LOWER与VALID

count\_include\_pad：int，暂不支持

kernel\_shape：可选，包括：

− kernel\_shape\[0\]：数据类型：int32，指定沿H维度的窗口大小，取值范围为\[1, 32768\]，默认为1

− kernel\_shape\[1\]：数据类型：int32，指定沿W维度的窗口大小，取值范围为\[1, 32768\]，默认为1

strides：可选，包括：

− strides\[0\]：数据类型：int32，指定沿H维度的步长，默认为1

− strides\[1\]：数据类型：int32，指定沿W维度的步长，默认为1

pads：可选，包括：

− pads\[0\]：数据类型：int32，指定顶部padding，默认为0

− pads\[1\]：数据类型：int32，指定底部padding，默认为0

− pads\[2\]：数据类型：int32，指定左部padding，默认为0

− pads\[3\]：数据类型：int32，指定右部padding，默认为0

ceil\_mode：可选，数据类型：int32，取值：0（floor模式），1（ceil模式），默认为0

【约束】

strides\[0\]或者strides\[1\]取值步长大于63时，会使用AI CPU计算，性能会下降；

kernel\_shape\_H或kernel\_shape\_W取值超过\[1,255\]，或者kernel\_shape\_H \* kernel\_shape\_W \> 256时，会使用AI CPU计算，导致性能下降；

1 <= input\_w <= 4096；

当输入张量的N是一个质数时，N应当小于65535；

ceil\_mode参数仅在auto\_pad='NOTSET'时生效；

不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时fp32类型输入；

auto\_pad属性值SAME\_UPPER, SAME\_LOWER统一使用的TBE的SAME属性，即TBE算子没有根据这个属性区分pad的填充位置，可能会带来精度问题

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="BatchNormalization.md">BatchNormalization</h2>

## 功能<a name="section12725193815114"></a>

标准化张量

## 边界<a name="section9981612134"></a>

【输入】

五个输入

X：数据类型为float16、float32的4D张量

scale：数据类型为float32的张量，指定尺度因子

B：数据类型为float32的张量，指定偏移量

mean：数据类型为float32的张量，指定均值

var：数据类型为float32的张量，指定方差

【输出】

五个输出

Y：标准化之后的张量，数据类型为float16或float32

mean：均值

var：方差

saved\_mean：在训练过程中使用已保存的平均值来加快梯度计算

saved\_var：在训练过程中使用已保存的方差来加快梯度计算

【属性】

epsilon：可选，数据类型：float32，指定一个小值与var相加，以避免除以0，默认为0.0001

momentum：float32，该参数暂不支持

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="BatchMatMul.md">BatchMatMul</h2>

## 功能<a name="section12725193815114"></a>

将两个输入执行矩阵乘

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16，float，int32

x2：一个tensor，数据类型：float16，float，int32

【属性】

两个属性：

adj\_x1：bool型

adj\_x2：bool型

【输出】

一个输出

y：一个tensor，数据类型：float16，float，int32

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="BatchMultiClassNMS.md">BatchMultiClassNMS</h2>

## 功能<a name="section12725193815114"></a>

为输入boxes和输入score计算nms

## 边界<a name="section9981612134"></a>

【输入】

两个必选输入，两个可选输入

boxes：一个tensor，数据类型：float16

scores：一个tensor，数据类型：float16

clip\_window：一个tensor，数据类型：float16

num\_valid\_boxes：一个tensor，数据类型：int32

【属性】

六个属性：

score\_threshold：float型

iou\_threshold：float型

max\_size\_per\_class：int型

max\_total\_size：int型

change\_coordinate\_frame：bool型

transpose\_box：bool型

【输出】

四个输出

nmsed\_boxes：一个tensor，数据类型：float16

nmsed\_scores：一个tensor，数据类型：float16

nmsed\_classes：一个tensor，数据类型：float16

nmsed\_num：一个tensor，数据类型：float16

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="BitShift.md">BitShift</h2>

## 功能<a name="section421532641316"></a>

元素级位移算子

## 边界<a name="section143631030111310"></a>

【输入】

两个输入

x：一个tensor，表示被位移的输入

y：一个tensor，表示位移的数量

【输出】

z：一个tensor，表示位移后的结果

【属性】

direction：数据类型：string，必选，指定位移方向，取值范围："RIGHT"或者"LEFT"

【约束】

当direction="LEFT"时不支持UINT16，UIN32，UINT64

## 支持的ONNX版本<a name="section098583811132"></a>

Opset v11/v12/v13

<h2 id="Cast.md">Cast</h2>

## 功能<a name="section12725193815114"></a>

将输入数据的type转换为指定的type

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor

【输出】

y：一个tensor，输出的数据类型为属性指定的类型，数据类型：bool、float16、float32、int8、int32、uint8等

【属性】

to：数据类型：int，必选，指定目标数据类型，取值范围：在指定的数据类型范围内

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Ceil.md">Ceil</h2>

## 功能<a name="section12725193815114"></a>

对输入张量向上取整

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Celu.md">Celu</h2>

## 功能<a name="section12725193815114"></a>

连续可微的指数线性单位：对输入张量X按元素执行线性单位，使用公式：

max\(0,x\) + min\(0,alpha\*\(exp\(x/alpha\)-1\)\)

## 边界<a name="section9981612134"></a>

【输入】

X：tensor\(float\)

【输出】

Y：tensor\(float\)

【属性】

alpha：float，默认值：1.0

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v12/v13

<h2 id="Concat.md">Concat</h2>

## 功能<a name="section12725193815114"></a>

对多个张量Concat

## 边界<a name="section9981612134"></a>

【输入】

inputs：多个输入张量，数据类型：float16、float32、int32、uint8、int16、int8、int64、qint8、quint8、qint32、uint16、uint32、uint64、qint16、quint16

【输出】

concat\_result：张量，与输入张量类型一致

【属性】

axis：指定哪一个轴进行concat操作，负数表示从后往前对维度计数，取值范围为\[-r, r - 1\]，r=rank\(inputs\)

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Clip.md">Clip</h2>

## 功能<a name="section12725193815114"></a>

将张量值剪辑到指定的最小值和最大值之间

## 边界<a name="section9981612134"></a>

【输入】

三个输入

X ：一个张量，数据类型：float16、float32、int32

min：一个scalar

max：一个scalar

【输出】

一个输出

Y：一个张量，剪辑后的输出，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ConvTranspose.md">ConvTranspose</h2>

## 功能<a name="section12725193815114"></a>

转置卷积

## 边界<a name="section9981612134"></a>

【输入】

3个输入

x：tensor，数据类型：float16、float32

w：tensor，数据类型：float16、float32

b：可选tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【属性】

auto\_pad：str，默认为NOTSET，含义：显式使用padding的方式

dilations：ints，默认为全1序列，含义：filter的每轴空洞值

group：int，默认为1，含义：输入通道分组数

kernel\_shape：ints，默认为w，含义：卷积核大小

output\_padding：ints，默认为全0数组，含义：指定padding值

output\_shape：ints，根据pad自动计算，含义：输出shape

pads：ints，默认为全0矩阵，含义：每根轴指定pad值

strides：ints，默认为全1矩阵，含义：每根轴的stride值

【约束】

目前只支持2D的转置卷积，3D及以上暂不支持

dilations只支持1

output\_shape支持限制：实现部分功能。现在支持output shape的大小，小于原始输入大小，但是不支持大于原始输入大小

算子不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时fp32，fp64的输入

属性auto\_pad不支持 "SAME\_UPPER"，"SAME\_LOWER"

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Cumsum.md">Cumsum</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量在给定axis上面的累加和

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x：一个tensor，数据类型：float16、float32、int32

axis：一个int32或者int64的标量，默认为0，范围为\[-rank\(x\), rank\(x\)-1\]

【输出】

一个输出

y：一个张量，和输入x同样的type

【属性】

exclusive：int，默认为0，含义：是否返回不包括顶层元素的和

reverse：int，默认为0，含义：是否反方向求和

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Conv.md">Conv</h2>

## 功能<a name="section12725193815114"></a>

卷积

## 边界<a name="section9981612134"></a>

【输入】

X：输入4D张量

W：权重张量

B：可选，偏差，一维张量

【输出】

Y：卷积输出张量

【属性】

auto\_pad：可选，支持VALID、NOTSET

dilations：4个整数的列表，指定用于扩张卷积的扩张率，H和W维度取值范围为\[1, 255\]

group：从输入通道到输出通道的阻塞连接数，输入通道和输出通道都必须被“group”整除；数据类型为int32，必须设置为1

pads：4个整数的列表，指定顶部、底部、左侧和右侧填充，取值范围为\[0, 255\]

strides：4个整数的列表，指定沿高度H和宽度W的卷积步长。H和W维度取值范围为\[1, 63\]，默认情况下，N和C尺寸设置为1

【约束】

输入张量，W维度取值范围为\[1, 4096\]

权重张量，H维度和W维度取值范围为\[1, 255\]

当输出张量的W == 1且H == 1时，输入张量和权重的H和W维度需相同

当输出张量的W = 1，H != 1时，算子不支持

不支持atc工具--precision\_mode=must\_keep\_origin\_dtype参数时输入类型为fp32和fp64

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="Compress.md">Compress</h2>

## 功能<a name="section12725193815114"></a>

按指定轴进行切片。

## 边界<a name="section9981612134"></a>

【输入】

两个输入：

input：维度大于等于1的tensor，支持类型：uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float, string, bool

condition：1维tensor，用于指定切片和需要选择的元素，支持类型：bool

【输出】

一个输出

output：tensor，类型：与输入一致

【属性】

axis：可选，int类型，进行切片的轴，如果没有指定轴，在切片之前将输入tensor展平。取值范围是\[-r,r-1\]，r为输入tensor的维数。

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v9//v11/v12/v13

<h2 id="Constant.md">Constant</h2>

## 功能<a name="section12725193815114"></a>

构建constant节点张量

## 边界<a name="section9981612134"></a>

【输入】

无

【输出】

一个输出

Y：输出张量，和提供的tensor值一致

【属性】

value：输出张量的值

【约束】

sparse\_value：不支持

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ConstantOfShape.md">ConstantOfShape</h2>

## 功能<a name="section12725193815114"></a>

用给定的值和shape生成张量

## 边界<a name="section9981612134"></a>

【输入】

x：1D的int64的tensor，表示输出数据的shape，所有的值必须大于0

【输出】

y：一个tensor，shape由输入指定，如果属性value指定了值，那输出的值和数据类型就等于value指定的值，如果属性value不指定，输出tensor的值默认为0，数据类型默认为float32

【属性】

value：指定输出tensor的数据和类型

【约束】

x：1<=len\(shape\)<=8

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="Cos.md">Cos</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的余弦值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Cosh.md">Cosh</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的双曲余弦

## 边界<a name="section9981612134"></a>

【输入】

一个输入

X1：一个tensor，数据类型：float16、float、double

【输出】

一个输出

y：一个张量，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="DeformableConv2D.md">DeformableConv2D</h2>

## 功能<a name="section421532641316"></a>

形变卷积

## 边界<a name="section143631030111310"></a>

【输入】

X：输入4D张量

filter：权重张量

offsets：偏移量，4维张量

bias：可选，偏差，一维张量

【输出】

Y：形变卷积输出张量

【属性】

auto\_pad：可选，支持VALID、NOTSET

dilations：4个整数的列表，指定用于扩张卷积的扩张率，H和W维度取值范围为\[1, 255\]

group：从输入通道到输出通道的阻塞连接数，输入通道和输出通道都必须被“group”整除；数据类型为int32，必须设置为1

pads：4个整数的列表，指定顶部、底部、左侧和右侧填充，取值范围为\[0, 255\]

strides：4个整数的列表，指定沿高度H和宽度W的卷积步长。H和W维度取值范围为\[1, 63\]，默认情况下，N和C尺寸设置为1

data\_format：string，表示输入数据format，默认是“NHWC”

deformable\_groups：分组卷积通道数，缺省为1

modulated：bool，指定DeformableConv2D版本，true表示v2版本，false表示v1版本，当前只支持true

【限制】

输入张量，W维度取值范围为\[1, 4096 / filter\_width\]，H取值范围为\[1, 100000 / filter\_height\]

权重张量，W维度取值范围为\[1, 63\]，H取值范围为\[1, 63\]

不支持atc工具--precision\_mode=must\_keep\_origin\_dtype参数时输入类型为fp32和fp64

## 支持的ONNX版本<a name="section19647924181413"></a>

自定义算子，无对应onnx版本

<h2 id="Det.md">Det</h2>

## 功能<a name="section12725193815114"></a>

计算方形矩阵行列式

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="DepthToSpace.md">DepthToSpace</h2>

## 功能<a name="section12725193815114"></a>

将数据由深度重排到空间数据块

## 边界<a name="section9981612134"></a>

【输入】

1个输入

input：format为NCHW的tensor输入，类型：float16、float32,double，int32,int64等

【输出】

1个输出

output：一个张量,shape为\[N, C/\(blocksize \* blocksize\), H \* blocksize, W \* blocksize\]

【属性】

blocksize：int，必选 指定被移动的块的大小

mode： string 指定是depth-column-row还是column-row-depth排列，默认DCR

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Div.md">Div</h2>

## 功能<a name="section12725193815114"></a>

按元素进行除法运算

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16、float32、double、int32、int64

x2：一个tensor，数据类型：float16、float32、double、int32、int64

【输出】

一个输出

y：一个tensor，数据类型和输入一致

【约束】

输入、输出的type相同

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Dropout.md">Dropout</h2>

## 功能<a name="section12725193815114"></a>

拷贝或者屏蔽输入数据

## 边界<a name="section9981612134"></a>

【输入】

1-3个输入

data：tensor输入，类型：float16、float32,double等

ratio：可选输入，类型：float16、float32,double等

training\_mode：可选输入，类型：bool

【输出】

1-2个输出

output：一个张量

mask： 一个张量

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Elu.md">Elu</h2>

## 功能<a name="section12725193815114"></a>

Elu激活函数

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【属性】

alpha：float，默认为1.0，含义：系数

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="EmbeddingBag.md">EmbeddingBag</h2>

## 功能<a name="section12725193815114"></a>

计算embedding函数的反向输出

## 边界<a name="section9981612134"></a>

【输入】

两个必选输入，两个可选输入

weight：一个tensor，数据类型：float32

indices：一个tensor，数据类型：int32

offset：一个tensor，数据类型：int32

per\_sample\_weights：一个tensor，数据类型：float32

【属性】

四个属性：

mode：string型

scale\_grad\_by\_fraq：bool型

sparse：bool型

include\_last\_offset：bool型

【输出】

一个输出

y：一个tensor，数据类型：float32

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="Equal.md">Equal</h2>

## 功能<a name="section12725193815114"></a>

判断两个输入张量对应位置是否相等

## 边界<a name="section9981612134"></a>

【输入】

两个输入

X1：一个tensor

X2：一个tensor

【输出】

一个输出

y：一个tensor ，数据类型：bool

【约束】

输入X1、X2的数据类型和格式相同，支持如下数据类型：bool、uint8、int8、int16、int32、int64、float16、float32、double

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Erf.md">Erf</h2>

## 功能<a name="section12725193815114"></a>

高斯误差函数

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型和格式与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="Exp.md">Exp</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的指数

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Expand.md">Expand</h2>

## 功能<a name="section12725193815114"></a>

将输入tensor广播到指定shape

## 边界<a name="section9981612134"></a>

【输入】

2个输入

input：tensor，数据类型：float16、float32

shape：tensor，数据类型：int64

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【约束】

需要修改模型将输入shape由placeholder改为const类型，可以使用onnxsimplifier简化模型

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="EyeLike.md">EyeLike</h2>

## 功能<a name="section421532641316"></a>

生成一个2D矩阵，主对角线是1，其他为0

## 边界<a name="section143631030111310"></a>

【输入】

1个输入

x：2维tensor，用于拷贝tensor的shape

【输出】

一个输出

y：一个张量，和输入x同样的shape

【属性】

dtype：int，指定输出数据类型

k：int，默认是0，表示主对角线被广播成1的索引。如y是输出，则y\[i, i+k\] = 1

【约束】

仅支持k=0

## 支持的ONNX版本<a name="section19647924181413"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Flatten.md">Flatten</h2>

## 功能<a name="section12725193815114"></a>

将张量展平

## 边界<a name="section9981612134"></a>

【输入】

input：多维张量，数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32

【输出】

具有输入张量的内容的2D张量

【属性】

axis：int，该参数暂不支持负值索引

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Floor.md">Floor</h2>

## 功能<a name="section12725193815114"></a>

对输入张量向下取整

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Gather.md">Gather</h2>

## 功能<a name="section12725193815114"></a>

根据相应的轴从“x”中收集切片

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16、float32、int32、int64、int8、int16、uint8、uint16、uint32、uint64、bool

indices：一个tensor，数据类型：int32、int64

【输出】

一个输出

y：一个张量，数据类型和输入x1类型一致

【属性】

axis：数据类型：int，指定gather的轴，取值范围为\[-r, r-1\]（r表示输入数据的秩）

【约束】

不支持indices为负值的索引

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="GatherND.md">GatherND</h2>

## 功能<a name="section12725193815114"></a>

将输入数据切片输出

## 边界<a name="section9981612134"></a>

【输入】

2个输入

data：秩r\>=1的tensor输入，类型：float16, float32, double, int32, int64等

indices：int64的索引张量,秩q\>=1

【输出】

1个输出

output：一个张量, 秩为q + r - indices\_shape\[-1\] - 1

【属性】

batch\_dims：int，默认为0 批处理轴的数量

【约束】

不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时double的输入

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v11/v12/v13

<h2 id="GatherElements.md">GatherElements</h2>

## 功能<a name="section12725193815114"></a>

获取索引位置的元素产生输出

## 边界<a name="section9981612134"></a>

【输入】

2个输入

input：秩大于1的tensor输入，类型：float16、float32,double，int32,int64等

indices：int32/int64的索引张量

【输出】

1个输出

output：一个张量,与indices的shape相同

【属性】

axis：int，默认为0 指定聚集的轴

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Gemm.md">Gemm</h2>

## 功能<a name="section12725193815114"></a>

通用矩阵乘

## 边界<a name="section9981612134"></a>

【输入】

A：2D矩阵张量，数据类型：float16、float32

B：2D矩阵张量，数据类型：float16、float32

C：偏差，可选，该参数暂不支持

【输出】

Y：2D矩阵张量，数据类型：float16、float32

【属性】

transA：bool，是否A需要转置

transB：bool，是否B需要转置

alpha：float，该参数暂不支持

beta：float，该参数暂不支持

【约束】

v8/v9/v10版本不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时fp32类型输入

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="GlobalAveragePool.md">GlobalAveragePool</h2>

## 功能<a name="section12725193815114"></a>

全局平均池化

## 边界<a name="section9981612134"></a>

【输入】

X：一个张量，数据类型：float16、float32，格式为NCHW

【输出】

Y：池化输出张量，数据类型与X相同，格式为NCHW

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="GlobalLpPool.md">GlobalLpPool</h2>

## 功能<a name="section12725193815114"></a>

全局范数池化算子

## 边界<a name="section9981612134"></a>

【输入】

2个输入

input：tensor，数据类型：float16、float32

p：可选属性， int32，默认2

【输出】

1个输出

y：更新后的张量数据，数据类型和输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="GlobalMaxPool.md">GlobalMaxPool</h2>

## 功能<a name="section12725193815114"></a>

全局最大池化算子

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：前一个节点的输出tensor，类型：float16, float32, double

【输出】

1个输出

output：池化后的张量

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Greater.md">Greater</h2>

## 功能<a name="section12725193815114"></a>

按元素比较输入x1和x2的大小，若x1\>x2，对应位置返回true

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16、float32、int32、int8、uint8

x2：一个tensor，数据类型：float16、float32、int32、int8、uint8

【输出】

一个输出

y：一个tensor，数据类型：bool

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="GreaterOrEqual.md">GreaterOrEqual</h2>

## 功能<a name="section12725193815114"></a>

按元素比较输入x1和x2的大小，若x1\>=x2，对应位置返回true

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16、float32、int32、int8、uint8等

x2：一个tensor，数据类型：float16、float32、int32、int8、uint8等

【输出】

一个输出

y：一个tensor，数据类型：bool

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v12

<h2 id="HardSigmoid.md">HardSigmoid</h2>

## 功能<a name="section12725193815114"></a>

HardSigmoid接受一个输入数据\(张量\)并生成一个输出数据\(张量\)，HardSigmoid函数y = max\(0, min\(1, alpha \* x + beta\)\)应用于张量元素方面。

## 边界<a name="section9981612134"></a>

【输入】

1个输入

X：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

【输出】

1个输出

Y：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

【属性】

alpha：float，默认值：0.2

beta：float，默认值：0.2

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v1/v6/v8/v9/v10/v11/v12/v13

<h2 id="hardmax.md">hardmax</h2>

## 功能<a name="section12725193815114"></a>

计算hardmax结果，如果元素是指定axis的最大元素则设为1，否则为0

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，rank=2，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【属性】

axis：int，默认为-1，含义：指定计算轴

【约束】

使用atc工具--precision\_mode参数必须为allow\_fp32\_to\_fp16

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="HardSwish.md">HardSwish</h2>

## 功能<a name="section12725193815114"></a>

HardSwish激活函数。y=x \* max\(0, min\(1, alpha \* x + beta \)\)，其中alpha=1/6，beat=0.5

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：tensor，数据类型：float16、float32

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v14

<h2 id="Identity.md">Identity</h2>

## 功能<a name="section12725193815114"></a>

恒等操作

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="If.md">If</h2>

## 功能<a name="section12725193815114"></a>

逻辑控制判断算子

## 边界<a name="section9981612134"></a>

【输入】

一个输入

cond：If op的条件

两个属性

else\_branch：条件为假的分支

then\_branch：条件为真的分支

【输出】

一到多个输出

y：tensor或者tensor序列

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="InstanceNormalization.md">InstanceNormalization</h2>

## 功能<a name="section421532641316"></a>

计算y = scale \* \(x - mean\) / sqrt\(variance + epsilon\) + B，其中mean 和 variance 是每个实例每个通道的均值和方法

## 边界<a name="section143631030111310"></a>

【输入】

3个输入

x： tensor，数据类型是float16，float

scale：1维tensor，维度同x的C轴长度，和输入x同样的dtype

B：1维tensor，维度同x的C轴长度，和输入x同样的dtype

【输出】

一个输出

y：一个张量，和输入x同样的shape和dtype

【属性】

epsilon：float，默认是1e-05，避免除0

## 支持的ONNX版本<a name="section19647924181413"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Less.md">Less</h2>

## 功能<a name="section12725193815114"></a>

按元素比较输入x1和x2的大小，若x1<x2，对应位置返回true

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16、float32、int32、int8、uint8

x2：一个tensor，数据类型：float16、float32、int32、int8、uint8

【输出】

一个输出

y：一个tensor，数据类型：bool

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="LeakyRelu.md">LeakyRelu</h2>

## 功能<a name="section12725193815114"></a>

对输入张量用leakrelu函数激活

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y： 一个tensor，数据类型和shape与输入一致

【属性】

alpha：数据类型为float，默认0.01，表示leakage系数

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="LessOrEqual.md">LessOrEqual</h2>

## 功能<a name="section12725193815114"></a>

小于等于计算

## 边界<a name="section9981612134"></a>

【输入】

2个输入

x：tensor，数据类型：float16、float32

y：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的shape,数据类型：bool

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v12/v13

<h2 id="Log.md">Log</h2>

## 功能<a name="section12725193815114"></a>

计算输入的自然对数

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="LogSoftMax.md">LogSoftMax</h2>

## 功能<a name="section12725193815114"></a>

对输入张量计算logsoftmax值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

【属性】

axis：数据类型为int；指定计算的轴，取值范围：\[-r, r-1\]，r为输入的秩

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="LpNormalization.md">LpNormalization</h2>

## 功能<a name="section12725193815114"></a>

给定一个矩阵，沿给定的轴应用LpNormalization。

## 边界<a name="section9981612134"></a>

【输入】

1个输入

input，类型：tensor\(float16\), tensor\(float\)

【输出】

1个输出

output，类型：tensor\(float16\), tensor\(float\)

【属性】

axis：int，默认值：-1

p：int，默认值：2

【约束】

auto\_pad属性值SAME\_UPPER, SAME\_LOWER统一使用的TBE的SAME属性，即TBE算子没有根据这个属性区分pad的填充位置，可能会带来精度问题

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v1/v8/v9/v10/v11/v12/v13

<h2 id="LpPool.md">LpPool</h2>

## 功能<a name="section12725193815114"></a>

Lp范数池化。

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：tensor，数据类型：float16

【输出】

一个输出

y：tensor，数据类型：float16

【属性】

auto\_pad：string，默认为NOTSET，支持：NOTSET, SAME\_UPPER, SAME\_LOWER 或者 VALID

kernel\_shape：必选，int列表，kernel每个轴上的尺寸

p：int，范数，默认为2

pads：int列表

strides：int列表

【约束】

auto\_pad属性值SAME\_UPPER, SAME\_LOWER统一使用的TBE的SAME属性，即TBE算子没有根据这个属性区分pad的填充位置，可能会带来精度问题

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v11/v12/v13

<h2 id="LRN.md">LRN</h2>

## 功能<a name="section12725193815114"></a>

对输入张量做局部响应归一化

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和format

【属性】

alpha：float，缩放因子

beta：float，指数项

bias：float

size：int，求和的通道数，只支持奇数

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="LSTM.md">LSTM</h2>

## 功能<a name="section12725193815114"></a>

计算单层LSTM。这个操作符通常通过一些自定义实现\(如CuDNN\)来支持。

## 边界<a name="section9981612134"></a>

【输入3-8】

X：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

W：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

R：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

B：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

sequence\_lens：，类型：tensor\(int32\)

initial\_h：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

initial\_c：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

p：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

【输出0-3】

Y：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

Y\_h：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

Y\_c：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

【属性】

activation\_alpha：list of floats

activation\_beta：list of floats

activations：list of strings

clip： float

direction： string，默认值：forward

hidden\_size： int

input\_forget： int，默认值：0

layout： int，默认值：0

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="MatMul.md">MatMul</h2>

## 功能<a name="section12725193815114"></a>

矩阵乘

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个2D的tensor，数据类型：float16

x2：一个2D的tensor，数据类型：float16

【输出】

一个输出

y：一个2D的tensor，数据类型：float16

【约束】

仅支持1-6维输入

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Max.md">Max</h2>

## 功能<a name="section12725193815114"></a>

元素级比较输入tensor的大小

## 边界<a name="section9981612134"></a>

【输入】

多个输入\(1-∞\)

data\_0：tensor的列表，类型：float16、float32,int8,int16,int32等

【输出】

一个输出

max：一个张量，和输入x同样的type和shape（广播后的shape）

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="MaxPool.md">MaxPool</h2>

## 功能<a name="section12725193815114"></a>

最大池化

## 边界<a name="section9981612134"></a>

【输入】

X：一个张量，数据类型：float16、float32，格式为NCHW

【输出】

Y：一个张量，数据类型：float16、float32，格式为NCHW

【属性】

auto\_pad：可选，支持SAME\_UPPER、SAME\_LOWER、VALID、NOTSET

storage\_order：暂不支持该参数

kernel\_shape：可选，包括：

-   kernel\_shape\[0\]：数据类型：int32，指定沿H维度的窗口大小，取值范围为\[1, 32768\]，默认为1
-   kernel\_shape\[1\]：数据类型：int32，指定沿W维度的窗口大小，取值范围为\[1, 32768\]，默认为1

strides：可选，包括：

-   strides\[0\]：数据类型：int32，指定沿H维度的步长，默认为1
-   strides\[1\]：数据类型：int32，指定沿W维度的步长，默认为1

pads：可选，包括：

-   pads\[0\]：数据类型：int32，指定顶部padding，默认为0
-   pads\[1\]：数据类型：int32，指定底部padding，默认为0
-   pads\[2\]：数据类型：int32，指定左部padding，默认为0
-   pads\[3\]：数据类型：int32，指定右部padding，默认为0

ceil\_mode：可选，数据类型：int32，取值：0\(floor模式），1（ceil模式），默认为0

【约束】

strides\[0\]或者strides\[1\]取值步长大于63时，会使用AI CPU计算，性能会下降；

kernel\_shape\_H或kernel\_shape\_W取值超过\[1,255\]，或者kernel\_shape\_H \* kernel\_shape\_W \> 256时，会使用AI CPU计算，导致性能下降；

1 <= input\_w <= 4096

当输入张量的N是一个质数时，N应小于65535

2D tensor输入不支持dilations

auto\_pad属性是VALID时，ceil\_mode属性值必须为0

不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时fp32类型输入

pads属性和auto\_pad属性不可同时使用

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="MaxRoiPool.md">MaxRoiPool</h2>

## 功能<a name="section12725193815114"></a>

ROI最大池消耗一个输入张量X和感兴趣区域\(ROI\)，以便在每个ROI上应用最大池，从而产生输出的4-D形状张量\(num\_roi, channels, pooled\_shape\[0\]， pooled\_shape\[1\]\)。

## 边界<a name="section9981612134"></a>

【输入】

X：，类型：tensor\(float16\), tensor\(float\)

rois：，类型：tensor\(float16\), tensor\(float\)

【输出】

Y：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

【属性】

pooled\_shape： list of ints

spatial\_scale： float，默认值：1.0

【约束】

不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时fp32类型输入

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/13

<h2 id="Mean.md">Mean</h2>

## 功能<a name="section12725193815114"></a>

每个输入张量的元素均值\(支持numpy风格的广播\)。所有输入和输出必须具有相同的数据类型。该操作符支持多向\(即numpy风格\)广播。

## 边界<a name="section9981612134"></a>

【输入1-∞】

data\_0：，类型：tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

mean：，类型：tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="MeanVarianceNormalization.md">MeanVarianceNormalization</h2>

## 功能<a name="section12725193815114"></a>

使用公式对输入张量X进行均值方差标准化：\(X-EX\)/sqrt\(E\(X-EX\)^2\)

## 边界<a name="section9981612134"></a>

【输入】

X：，类型：tensor\(float16\), tensor\(float\), tensor\(bfloat16\)

【输出】

Y：，类型：tensor\(float16\), tensor\(float\), tensor\(bfloat16\)

【属性】

axes： list of ints，默认值：\['0', '2', '3'\]

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="Min.md">Min</h2>

## 功能<a name="section12725193815114"></a>

计算输入tensors的最小值

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor列表，数据类型：float16、float32

【输出】

一个输出

y：计算出最小值的tensor

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Mod.md">Mod</h2>

## 功能<a name="section12725193815114"></a>

执行元素二进制模数\(支持numpy风格的广播\)。余数的符号与除数的符号相同。

## 边界<a name="section9981612134"></a>

【输入】

A：tensor\(uint8\), tensor\(uint16\), tensor\(uint32\), tensor\(uint64\), tensor\(int8\), tensor\(int16\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

B：tensor\(uint8\), tensor\(uint16\), tensor\(uint32\), tensor\(uint64\), tensor\(int8\), tensor\(int16\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

C：，类型：tensor\(uint8\), tensor\(uint16\), tensor\(uint32\), tensor\(uint64\), tensor\(int8\), tensor\(int16\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【属性】

fmod：int，默认值：0

【约束】

当输入类型为浮点时，fmod不支持为0

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12/v13

<h2 id="Mul.md">Mul</h2>

## 功能<a name="section12725193815114"></a>

矩阵点乘

## 边界<a name="section9981612134"></a>

【输入】

A：一个张量，数据类型：float16、float32、uint8、int8、int16、int32

B：一个张量，数据类型：float16、float32、uint8、int8、int16、int32

【输出】

C：一个张量，数据类型与输入张量一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Multinomial.md">Multinomial</h2>

## 功能<a name="section12725193815114"></a>

返回Multinomial采样结果矩阵

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，shape=\[batch\_size, class\_size\]，数据类型：float16、float32

【输出】

一个输出

y：一个张量，shape=\[batch\_size, sample\_size\]，输出type是int32、int64

【属性】

dtype：int，默认为6，含义：输出dtype，默认为int32

sample\_size：int，默认为1，含义：采样次数

seed：float，随机数种子

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Neg.md">Neg</h2>

## 功能<a name="section12725193815114"></a>

求输入的负数

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、int32

【输出】

一个输出

y：一个tensor，数据类型与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="NonMaxSuppression.md">NonMaxSuppression</h2>

## 功能<a name="section12725193815114"></a>

过滤掉与先前选定的框有较高重叠的“交集-并集”\(IOU\)框。移除得分小于score\_threshold的边界框。边界框格式由属性center\_point\_box表示。注意，该算法不知道原点在坐标系中的位置，更普遍地说，它对坐标系的正交变换和平移是不变的;因此，平移或反射坐标系统的结果在相同的方框被算法选择。selected\_indices输出是一组整数，索引到表示所选框的边界框的输入集合中。然后，可以使用Gather或gathernd操作获得与所选索引对应的边框坐标。

## 边界<a name="section9981612134"></a>

【输入2-5】

boxes： tensor\(float\)

scores： tensor\(float\)

max\_output\_boxes\_per\_class： 可选，数据类型：tensor\(int64\)

iou\_threshold： 可选，数据类型：tensor\(float\)

score\_threshold： 可选，数据类型：tensor\(float\)

【输出】

selected\_indices： tensor\(int64\)

【属性】

center\_point\_box： int 默认值：0

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12/v13

<h2 id="NonZero.md">NonZero</h2>

## 功能<a name="section12725193815114"></a>

返回非零元素的索引（行主序）

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、int32、int8、uint8等

【输出】

一个输出

y：一个tensor，数据类型：int64

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="Not.md">Not</h2>

## 功能<a name="section12725193815114"></a>

逻辑非

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：bool

【输出】

一个输出

y：一个tensor，数据类型：bool

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="OneHot.md">OneHot</h2>

## 功能<a name="section12725193815114"></a>

根据输入生成独热编码张量

## 边界<a name="section9981612134"></a>

【输入】

三个输入

indices：一个tensor，数据类型：uint8，uint16， uint32，uint64，int8，int16，int32，int64，float16，float，double

depth：一个tensor，数据类型：uint8，uint16， uint32，uint64，int8，int16，int32，int64，float16，float，double

values：一个tensor，数据类型：uint8，uint16， uint32，uint64，int8，int16，int32，int64，float16，float，double

【属性】

一个属性

axis：（可选）添加独热表示的轴

【输出】

一个输出

y：一个tensor，数据类型与values输入的类型一致

【约束】

算子属性不支持axis<-1

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

<h2 id="Or.md">Or</h2>

## 功能<a name="section12725193815114"></a>

逻辑或

## 边界<a name="section9981612134"></a>

【输入】

两个输入

X1：一个tensor，数据类型：bool

X2：一个tensor，数据类型：bool

【输出】

一个输出

y：一个tensor，数据类型：bool

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="RandomNormalLike.md">RandomNormalLike</h2>

## 功能<a name="section421532641316"></a>

根据正态分布生成随机数矩阵，输出tensor的shape与输入相同

## 边界<a name="section143631030111310"></a>

【输入】

1个输入

x： tensor，数据类型是float16，float

【输出】

一个输出

y：一个张量，和输入x同样的shape和dtype

【属性】

dtype：int，指定输出tensor的dtype

mean：float，默认是0.0，正态分布的均值

scale：float，默认是1.0，正态分布的标准差

seed：float，随机数种子

## 支持的ONNX版本<a name="section19647924181413"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="RandomUniformLike.md">RandomUniformLike</h2>

## 功能<a name="section421532641316"></a>

根据均匀分布生成随机数矩阵，输出tensor的shape与输入相同

## 边界<a name="section143631030111310"></a>

【输入】

1个输入

x：tensor，数据类型是float16，float

【输出】

一个输出

y：一个张量，和输入x同样的shape和dtype

【属性】

dtype：int，指定输出tensor的dtype

high：float，默认是1.0，均匀分布的上界

low：float，默认是0.0，均匀分布的下界

seed：float，随机数种子

## 支持的ONNX版本<a name="section19647924181413"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="RandomUniform.md">RandomUniform</h2>

## 功能<a name="section12725193815114"></a>

生成具有从均匀分布绘制的随机值的张量

## 边界<a name="section9981612134"></a>

【属性】

五个属性

dtype：int类型，指明输出类型

high：float型，指明上边界

low：float型，指明下边界

seed：\(可选\)，随机种子

shape：输出的形状

【输出】

一个输出

y：一个tensor，数据类型与dtype属性指定类型一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Range.md">Range</h2>

## 功能<a name="section12725193815114"></a>

产生一个连续序列的tensor

## 边界<a name="section9981612134"></a>

【输入】

3个输入

start：scalar，数据类型：float16、float32

limit：scalar，数据类型：float16、float32

delta：scalar，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Reciprocal.md">Reciprocal</h2>

## 功能<a name="section12725193815114"></a>

将输入张量取倒数

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ReduceL1.md">ReduceL1</h2>

## 功能<a name="section12725193815114"></a>

沿所提供的轴计算输入张量元素的L1范数。如果keepdim等于1，得到的张量的秩与输入的相同。如果keepdim等于0，那么得到的张量就会被精简维数。上述行为与numpy类似，只是numpy默认keepdim为False而不是True。

## 边界<a name="section9981612134"></a>

【输入】

data：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

reduced：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【属性】

axes： list of ints

keepdims： int，默认值：1

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ReduceL2.md">ReduceL2</h2>

## 功能<a name="section12725193815114"></a>

沿所提供的轴计算输入张量元素的L2范数。如果keepdim等于1，得到的张量的秩与输入的相同。如果keepdim等于0，那么得到的张量就会被精简维数。上述行为与numpy类似，只是numpy默认keepdim为False而不是True。

## 边界<a name="section9981612134"></a>

【输入】

data：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

reduced：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【属性】

axes： list of ints

keepdims： int，默认值：1

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ReduceLogSum.md">ReduceLogSum</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量指定方向的对数和

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16, float32

【输出】

一个输出

y：一个tensor，数据类型：float16, float32

【属性】

axes：数据类型为listInt；含义：指定计算轴；取值范围：\[-r, r-1\]，r是输入数据的维数

keepdims：数据类型为int；含义：是否保留缩减后的维度；默认为1

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v11/v13

<h2 id="ReduceLogSumExp.md">ReduceLogSumExp</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量指定方向的对数和的指数

## 边界<a name="section9981612134"></a>

【输入】

一个输入

data：一个tensor，数据类型：float16, float32

【输出】

一个输出

reduced：一个tensor，数据类型：float16, float32

【属性】

axes：一维tensor，数据类型int32、int64，含义：指定计算轴；取值范围：\[-r, r-1\]，r是输入数据的维数

keepdims：数据类型为int；含义：是否缩减维度；默认为1表示缩减维度

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ReduceMin.md">ReduceMin</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量指定方向的最小值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型：float16、float32

【属性】

axes：数据类型为listInt；含义：指定计算轴；取值范围：\[-r, r-1\]，r是输入数据的维数

keepdims：数据类型为int；含义：是否保留缩减后的维度；默认为1

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ReduceMean.md">ReduceMean</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的指定维度的元素的均值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和format

【属性】

axes：一个1D的整数列表，含义：指定精减的维度，取值范围为\[-r, r - 1\]，r是输入矩阵的秩

keepdims：数据类型为int，默认为1，含义：是否保留缩减后的维度

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ReduceProd.md">ReduceProd</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的元素沿所提供的轴的乘积。如果keepdim等于1，得到的张量的秩与输入的相同。如果keepdim等于0，那么得到的张量就会被精简维数。

## 边界<a name="section9981612134"></a>

【输入】

data：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

reduced：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【属性】

axes： list of ints

keepdims： int，默认值：1

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ReduceSumSquare.md">ReduceSumSquare</h2>

## 功能<a name="section12725193815114"></a>

沿所提供的轴计算输入张量元素的平方和。如果keepdim等于1，得到的张量的秩与输入的相同。如果keepdim等于0，那么得到的张量就会被精简维数。上述行为与numpy类似，只是numpy默认keepdim为False而不是True。

## 边界<a name="section9981612134"></a>

【输入】

data：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

reduced：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【属性】

axes： list of ints

keepdims： int，默认值：1

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v1/v8/v9/v10/v11/v12/v13

<h2 id="Resize.md">Resize</h2>

## 功能<a name="section12725193815114"></a>

调整输入tensor大小

## 边界<a name="section9981612134"></a>

【输入】

4个输入

x：一个tensor，数据类型：float16、float32

roi： 被输入图像归一化的1Dtensor，\[start1, ..., startN, end1, ..., endN\]，数据类型：float16、float32

scales：与输入x的秩相等的数组

sizes：输出tensor的size

【输出】

一个输出

y：缩放后的张量

【属性】

coordinate\_transformation\_mode：str，默认为half\_pixel，含义：定义缩放后图像与原图像的坐标转换

cubic\_coeff\_a：float，默认为-0.75，含义：三次插值系数

exclude\_outside：int，默认为0，含义：超出tensor外的权重

mode：str，默认为nearest，含义：插值算法，包括nearest, linear and cubic

nearest\_mode：str，默认为round\_prefer\_floor，含义：最近邻算子模式

【约束】

目前仅支持nearest和linear插值方式来处理图片，并且需要修改模型将输入scales或sizes由placeholder改为const类型，可以使用onnxsimplifier简化模型

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12

<h2 id="Relu.md">Relu</h2>

## 功能<a name="section12725193815114"></a>

整流线性单位函数

## 边界<a name="section9981612134"></a>

【输入】

X：输入张量，数据类型：float32、int32、uint8、int16、int8、uint16、float16、qint8

【输出】

Y：输出张量，数据类型与X一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ReduceSum.md">ReduceSum</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量指定维度的元素的和

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x的type和format相同

【属性】

axes：一个1D的整数列表，含义：指定精减的维度，取值范围为\[-r, r - 1\]（r是输入矩阵的秩）

keepdims：数据类型为int，默认为1，含义：是否保留缩减后的维度

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ReduceMax.md">ReduceMax</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量指定方向的最大值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、int32

【输出】

一个输出

y：一个tensor，数据类型：float16、float32、int32

【属性】

axes：数据类型为listInt；含义：指定计算轴；取值范围：\[-r, r-1\]，r是输入数据的秩

keepdims：数据类型为int；含义：是否保留缩减后的维度；默认为1

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Reshape.md">Reshape</h2>

## 功能<a name="section12725193815114"></a>

改变输入维度

## 边界<a name="section9981612134"></a>

【输入】

两个输入

data：一个张量

shape：一个张量，定义了输出张量的形状，int64

【输出】

reshaped：一个张量

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ReverseSequence.md">ReverseSequence</h2>

## 功能<a name="section12725193815114"></a>

根据指定长度对batch序列进行排序

## 边界<a name="section9981612134"></a>

【输入】

2个输入

x：tensor，rank \>= 2，数据类型：float16、float32

sequence\_lens：tensor，每个batch的指定长度，数据类型：int64

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【属性】

batch\_axis：int，默认为1，含义：指定batch轴

time\_axis：int，默认为1，含义：指定time轴

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12/v13

<h2 id="RoiExtractor.md">RoiExtractor</h2>

## 功能<a name="section12725193815114"></a>

从特征映射列表中获取ROI特征矩阵

## 边界<a name="section9981612134"></a>

【输入】

两个输入

features：一个tensor，数据类型：float32,float16

rois：一个tensor，数据类型：float32,float16

【属性】

八个属性：

finest\_scale：int型

roi\_scale\_factor：float型

spatial\_scale：float型数组

pooled\_height：int型

pooled\_width：int型

sample\_num：int型

pool\_mode：string型

aligned：bool型

【输出】

一个输出

y：一个tensor，数据类型：float32,float16

## 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

<h2 id="RoiAlign.md">RoiAlign</h2>

## 功能<a name="section12725193815114"></a>

在每个roi区域进行池化处理

## 边界<a name="section9981612134"></a>

【输入】

3个输入

x：tensor，4D输入，数据类型：float16、float32

rois：shape=\(num\_rois, 4\)，数据类型：float16、float32

batch\_indices ：shape=\(num\_rois,\)，数据类型：int64

【输出】

一个输出

y：一个张量，和输入x同样的type，shape=\(num\_rois, C, output\_height, output\_width\)

【属性】

mode：string，默认为avg，含义：池化方式

output\_height：int，默认为1，含义：y的高度

output\_width：int，默认为1，含义：y的宽度

sampling\_ratio ：int，默认为0，含义：插值算法采样点数

spatial\_scale：float，默认为1.0，含义：相对于输入图像的空间采样率

【约束】

batch\_indices数据类型只能写int32不能写int64

不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时fp32，fp64的输入

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12/v13

<h2 id="Round.md">Round</h2>

## 功能<a name="section12725193815114"></a>

对输入张量做四舍五入的运算

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="PRelu.md">PRelu</h2>

## 功能<a name="section12725193815114"></a>

PRelu激活函数

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x：一个tensor，数据类型：float16、float32

slope：slope张量，数据类型和输入x一致

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【约束】

slope必须是1维，当输入x的shape是1维时，slope的维度值必须为1；输入x的shape是其他维度时，slope的维度值可以为1或者为输入x的shape\[1\]

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Scatter.md">Scatter</h2>

## 功能<a name="section421532641316"></a>

根据updates和indices来更新data的值，并把结果返回。

## 边界<a name="section143631030111310"></a>

【输入】

3个输入

data： tensor，数据类型是float16，float，int32

indices：tensor，数据类型是int32、int64

updates：tensor，数据类型同data

【输出】

一个输出

y：一个张量，和输入x同样的shape和dtype

【属性】

axis：int，默认是0，表示沿axis取数据

## 支持的ONNX版本<a name="section19647924181413"></a>

Opset v9/v10

<h2 id="ScatterElements.md">ScatterElements</h2>

## 功能<a name="section421532641316"></a>

根据updates和indices来更新data的值，并把结果返回。

## 边界<a name="section143631030111310"></a>

【输入】

1个输入

data： tensor，数据类型是float16，float，int32

indices：tensor，数据类型是int32、int64

updates：tensor，数据类型同data

【输出】

一个输出

y：一个张量，和输入x同样的shape和dtype

【属性】

axis：int，默认是0，表示沿axis取数据

## 支持的ONNX版本<a name="section19647924181413"></a>

Opset v11/v12/v13

<h2 id="ScatterND.md">ScatterND</h2>

## 功能<a name="section12725193815114"></a>

创建data的拷贝，同时在指定indices处根据updates更新

## 边界<a name="section9981612134"></a>

【输入】

3个输入

data：tensor，rank \>= 1，数据类型：float16、float32

indices：tensor，rank \>= 1，数据类型：int64

updates：tensor，rank = q + r - indices\_shape\[-1\] - 1，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v11

<h2 id="Shrink.md">Shrink</h2>

## 功能<a name="section421532641316"></a>

单输入单输出计算，If x < -lambd, y = x + bias; If x \> lambd, y = x - bias; Otherwise, y = 0.

## 边界<a name="section143631030111310"></a>

【输入】

1个输入

data： tensor，数据类型是float16，float

【输出】

一个输出

y：一个张量，和输入x同样的shape和dtype

【属性】

bias：float，默认是0.0

lambda：float，默认是0.5

## 支持的ONNX版本<a name="section19647924181413"></a>

Opset v9/v10/v11/ v12/v13

<h2 id="Selu.md">Selu</h2>

## 功能<a name="section12725193815114"></a>

在元素级别使用指数线性单位函数y = gamma \* \(alpha \* e^x - alpha\) for x <= 0, y = gamma \* x for x \> 0 生成张量

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：fp16,fp32,double类型的tensor

两个属性

alpha：乘数因子

gamma：乘数因子

【输出】

一个输出

y：与输入类型相同的tensor

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Shape.md">Shape</h2>

## 功能<a name="section12725193815114"></a>

获取输入tensor的shape

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor

【输出】

y：输入tensor的shape，数据类型为int64的tensor

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Sigmoid.md">Sigmoid</h2>

## 功能<a name="section12725193815114"></a>

对输入做sigmoid

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：数据类型支持float16、float32

【输出】

一个输出

y：数据类型和输入x一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Slice.md">Slice</h2>

## 功能<a name="section12725193815114"></a>

获取输入tensor的切片

## 边界<a name="section9981612134"></a>

【输入】

五个输入

x：输入的tensor，数据类型：float16、float32、int32、uint8、bool、int8

starts：1Dtensor，int32或者int64，表示开始的索引位置

ends：1Dtensor，int32或者int64，表示结束的索引位置

axes：可选，1Dtensor，int32或者int64，表示切片的轴，取值范围为\[-r, r-1\]（r表示输入数据的秩）

steps：可选，1Dtensor，int32或者int64，表示切片的步长，最后一个轴的steps取值必须为1

【输出】

y：切片后的张量数据，数据类型和输入一致

【约束】

x：输入tensor维度不能为1

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Softmax.md">Softmax</h2>

## 功能<a name="section12725193815114"></a>

对输入进行softmax

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，类型和shape与输入x一致

【属性】

axis：Int，可选，表示进行softmax的方向，默认值为-1，范围为\[ -len\(x.shape\), len\(x.shape\)-1\]

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Softsign.md">Softsign</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的softsign\(x/\(1+|x|\)\)

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Softplus.md">Softplus</h2>

## 功能<a name="section12725193815114"></a>

计算softplus

## 边界<a name="section9981612134"></a>

【输入】

一个输入

X：1D的输入张量

【输出】

一个输出

Y：1D的张量

【约束】

数据类型仅支持float16、float32

输入、输出的数据类型一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="SpaceToDepth.md">SpaceToDepth</h2>

## 功能<a name="section12725193815114"></a>

SpaceToDepth将空间数据块重新排列成深度。更具体地说，这个op输出一个输入张量的副本，其中高度和宽度维度的值移动到深度维度。

## 边界<a name="section9981612134"></a>

【输入】

input：tensor\(uint8\), tensor\(uint16\), tensor\(uint32\), tensor\(uint64\), tensor\(int8\), tensor\(int16\), tensor\(int32\), tensor\(int64\), tensor\(bfloat16\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(string\), tensor\(bool\), tensor\(complex64\), tensor\(complex128\)

【输出】

output：tensor\(uint8\), tensor\(uint16\), tensor\(uint32\), tensor\(uint64\), tensor\(int8\), tensor\(int16\), tensor\(int32\), tensor\(int64\), tensor\(bfloat16\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(string\), tensor\(bool\), tensor\(complex64\), tensor\(complex128\)

【属性】

blocksize： int

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Split.md">Split</h2>

## 功能<a name="section12725193815114"></a>

将输入切分成多个输出

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、int8、int16、int32、int64、uint8、uint16、uint32、uint64

【输出】

一个输出

y：由多个输出tensor组成的列表，每个tensor数据类型和输入x一致

【属性】

split：list，数据类型：int8、int16、int32、int64，指定每个输出tensor沿着切分方向的大小

axis：数据类型：int8、int16、int32、int64，指定切分的方向

【约束】

split的每个元素必须\>=1

split的所有元素之和必须等于axis指定的切分方向的size

axis在\[ -len\(x.shape\), len\(x.shape\)-1\] 之间

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Sqrt.md">Sqrt</h2>

## 功能<a name="section12725193815114"></a>

计算元素的平方根

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor

【输出】

一个输出

y：一个tensor

【约束】

输入、输出的数据类型相同，支持的数据类型：float16、float32

如果x小于0，返回Nan

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Squeeze.md">Squeeze</h2>

## 功能<a name="section12725193815114"></a>

从输入中去除尺寸为1的维度

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个张量，数据类型：float16、float32、double、uint8、uint16、uint32、uint64、int8、int16、int32、int64、bool

【输出】

y：一个tensor，数据类型和输入一致

【属性】

axes：一个数据类型为int32或者int64的整形列表，维度为1；取值范围为\[-r, r-1\]（r表示输入张量的秩，负数表示从后面计算维度）；含义：指定要去除的维度

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Sub.md">Sub</h2>

## 功能<a name="section12725193815114"></a>

进行张量的减法运算

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor

x2：一个tensor

【输出】

一个输出

y：一个张量，数据类型和输入一致

【约束】

输入、输出的shape和dtype相同，支持的数据类型：int32、float16、float32

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Sign.md">Sign</h2>

## 功能<a name="section12725193815114"></a>

逐元素计算输入tensor的符号

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Sin.md">Sin</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量的正弦值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Sinh.md">Sinh</h2>

## 功能<a name="section12725193815114"></a>

计算输入张量双曲正弦值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Size.md">Size</h2>

## 功能<a name="section12725193815114"></a>

计算输入tensor的元素个数

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个int64的scalar

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Sum.md">Sum</h2>

## 功能<a name="section12725193815114"></a>

求和

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor序列，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Tanh.md">Tanh</h2>

## 功能<a name="section12725193815114"></a>

计算输入的双曲正切值

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型与输入一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="TfIdfVectorizer.md">TfIdfVectorizer</h2>

## 功能<a name="section421532641316"></a>

将输入文本序列向量化

## 边界<a name="section143631030111310"></a>

【输入】

1个输入

data： tensor，数据类型是int32，int64

【输出】

一个输出

y：一个张量，数据类型是float

【属性】

max\_gram\_length：int，最大n-gram长度

max\_skip\_count：int，从data中构造n-gram时最多skip数

min\_gram\_length：int，最小n-gram长度

mode：string，加权模式。可选为"TF" \(term frequency\), "IDF" \(inverse document frequency\)和"TFIDF" \(the combination of TF and IDF\)

ngram\_counts：int列表，n-gram池化的开始索引，有助于确认两个连续n-gram边界

ngram\_indexes：int列表，第i个元素表示输出tensor中第i个n-gram的坐标

pool\_int64s：int列表，不能与pool\_strings同时赋值，表示从训练集学到的n-grams

pool\_strings：string列表，与pool\_int64s含义一样。

weights：float列表，存储每个n-gram的池化权重数值

## 支持的ONNX版本<a name="section19647924181413"></a>

Opset v9/v10/v11/ v12/v13

<h2 id="Tile.md">Tile</h2>

## 功能<a name="section12725193815114"></a>

将输入张量沿指定维度重复

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x：一个tensor

repeats：一个1D的int64的tensor，size和输入的维度数一样

【输出】

一个输出

y：输出的tensor，type和维度与输入一致，output\_dim\[i\] = input\_dim\[i\] \* repeats\[i\]

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="ThresholdedRelu.md">ThresholdedRelu</h2>

## 功能<a name="section12725193815114"></a>

当x \> alpha时y = x，否则y=0

## 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【属性】

alpha：float，默认为1.0，含义：阈值

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12/v13

<h2 id="TopK.md">TopK</h2>

## 功能<a name="section12725193815114"></a>

返回指定轴的k个最大或最小值

## 边界<a name="section9981612134"></a>

【输入】

2个输入

x：tensor，数据类型：float16、float32

k：tensor，数据类型：int64

【输出】

2个输出

Values：topk的返回值

Indices：topk的返回值索引

【属性】

axis：int，默认为-1，含义：指定排序的轴

largest：int，默认为1，含义：返回k个最大/最小值

sorted：int，默认为1，含义：是否升序

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Transpose.md">Transpose</h2>

## 功能<a name="section12725193815114"></a>

转置

## 边界<a name="section9981612134"></a>

【输入】

data：一个张量，数据类型：float16、float32、int8、int16、int32、int64、uint8、uint16、uint32、uint64

【输出】

transposed：转置之后的张量

【属性】

perm：必选，整数列表， 张量data的维度排列

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Pad.md">Pad</h2>

## 功能<a name="section12725193815114"></a>

对输入tensor做填充

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x：数据类型支持float16、float32、int32

pads：数据类型支持int32 、int64

【输出】

一个输出

y：数据类型和输入x一致

【约束】

当mode值为constant时，目前仅支持constant\_value=0

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Pow.md">Pow</h2>

## 功能<a name="section12725193815114"></a>

计算输入x1的x2次幂

## 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16、float32、double、int32、int8、uint8

x2：一个tensor，数据类型和输入x1一致

【输出】

一个输出

y：数据类型和输入x1一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Unsqueeze.md">Unsqueeze</h2>

## 功能<a name="section12725193815114"></a>

在输入张量（数据）的形状中插入一维项

## 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：uint8、uint16、uint32、int8、int16、int32、float16、float32

【输出】

一个输出

y：一个tensor，数据类型和输入x一致

【属性】

axes：ListInt，表示在指定的维度进行插1维项，取值范围为\[-input\_rank, input\_rank\]，input\_rank为输入张量的秩，axes的内容不可以重复

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/10/v11/v12

<h2 id="Xor.md">Xor</h2>

## 功能<a name="section12725193815114"></a>

输入张量元素的xor逻辑运算

## 边界<a name="section9981612134"></a>

【输入】

两个输入

a：一个tensor，数据类型bool

b：一个tensor，数据类型bool

【输出】

c：一个tensor，数据类型bool

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

<h2 id="Where.md">Where</h2>

## 功能<a name="section12725193815114"></a>

根据条件从两个输入中选择元素

## 边界<a name="section9981612134"></a>

【输入】

三个输入

condition，条件，数据类型：bool

x：一个tensor，条件为true时从x中选取元素，数据类型支持float16、float32、int8、int32、uint8

y：一个tensor，条件为false时从y中选取元素，和x的数据类型一致

【输出】

一个tensor，数据类型和输入x一致

## 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

