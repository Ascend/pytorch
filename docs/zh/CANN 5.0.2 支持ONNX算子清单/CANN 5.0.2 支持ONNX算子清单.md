# ONNX算子规格清单








-   [Abs](#Abs)
-   [Acos](#Acos)
-   [Acosh](#Acosh)
-   [AdaptiveMaxPool2D](#AdaptiveMaxPool2D)
-   [Add](#Add)
-   [Addcmul](#Addcmul)
-   [AffineGrid](#AffineGrid)
-   [And](#And)
-   [Argmax](#Argmax)
-   [Argmin](#Argmin)
-   [AscendRequantS16](#AscendRequantS16)
-   [AscendRequant](#AscendRequant)
-   [AscendQuant](#AscendQuant)
-   [AscendDequantS16](#AscendDequantS16)
-   [AscendDequant](#AscendDequant)
-   [AscendAntiQuant](#AscendAntiQuant)
-   [Asin](#Asin)
-   [Asinh](#Asinh)
-   [Atan](#Atan)
-   [Atanh](#Atanh)
-   [AveragePool](#AveragePool)
-   [BatchNormalization](#BatchNormalization)
-   [BatchMatMul](#BatchMatMul)
-   [BatchMultiClassNMS](#BatchMultiClassNMS)
-   [Cast](#Cast)
-   [Ceil](#Ceil)
-   [Celu](#Celu)
-   [Concat](#Concat)
-   [Clip](#Clip)
-   [ConvTranspose](#ConvTranspose)
-   [Cumsum](#Cumsum)
-   [Conv](#Conv)
-   [Constant](#Constant)
-   [ConstantOfShape](#ConstantOfShape)
-   [Cos](#Cos)
-   [Cosh](#Cosh)
-   [Det](#Det)
-   [DepthToSpace](#DepthToSpace)
-   [Div](#Div)
-   [Dropout](#Dropout)
-   [elu](#elu)
-   [EmbeddingBag](#EmbeddingBag)
-   [Equal](#Equal)
-   [Erf](#Erf)
-   [Exp](#Exp)
-   [Expand](#Expand)
-   [Flatten](#Flatten)
-   [Floor](#Floor)
-   [Gather](#Gather)
-   [GatherND](#GatherND)
-   [GatherElements](#GatherElements)
-   [Gemm](#Gemm)
-   [GlobalAveragePool](#GlobalAveragePool)
-   [GlobalMaxPool](#GlobalMaxPool)
-   [Greater](#Greater)
-   [GreaterOrEqual](#GreaterOrEqual)
-   [Gru](#Gru)
-   [HardSigmoid](#HardSigmoid)
-   [hardmax](#hardmax)
-   [Identity](#Identity)
-   [If](#If)
-   [Less](#Less)
-   [LeakyRelu](#LeakyRelu)
-   [LessOrEqual](#LessOrEqual)
-   [Log](#Log)
-   [LogSoftMax](#LogSoftMax)
-   [LpNormalization](#LpNormalization)
-   [LRN](#LRN)
-   [LSTM](#LSTM)
-   [MatMul](#MatMul)
-   [Max](#Max)
-   [MaxPool](#MaxPool)
-   [MaxRoiPool](#MaxRoiPool)
-   [Mean](#Mean)
-   [MeanVarianceNormalization](#MeanVarianceNormalization)
-   [Min](#Min)
-   [Mod](#Mod)
-   [Mul](#Mul)
-   [Multinomial](#Multinomial)
-   [Neg](#Neg)
-   [NonMaxSuppression](#NonMaxSuppression)
-   [NonZero](#NonZero)
-   [Not](#Not)
-   [OneHot](#OneHot)
-   [Or](#Or)
-   [randomUniform](#randomUniform)
-   [Range](#Range)
-   [Reciprocal](#Reciprocal)
-   [ReduceL1](#ReduceL1)
-   [ReduceL2](#ReduceL2)
-   [ReduceMin](#ReduceMin)
-   [ReduceMean](#ReduceMean)
-   [ReduceProd](#ReduceProd)
-   [ReduceSumSquare](#ReduceSumSquare)
-   [Resize](#Resize)
-   [Relu](#Relu)
-   [ReduceSum](#ReduceSum)
-   [ReduceMax](#ReduceMax)
-   [Reshape](#Reshape)
-   [ReverseSequence](#ReverseSequence)
-   [RoiExtractor](#RoiExtractor)
-   [RoiAlign](#RoiAlign)
-   [Round](#Round)
-   [PRelu](#PRelu)
-   [ScatterND](#ScatterND)
-   [Selu](#Selu)
-   [Shape](#Shape)
-   [Sigmoid](#Sigmoid)
-   [Slice](#Slice)
-   [Softmax](#Softmax)
-   [Softsign](#Softsign)
-   [Softplus](#Softplus)
-   [SpaceToDepth](#SpaceToDepth)
-   [Split](#Split)
-   [Sqrt](#Sqrt)
-   [Squeeze](#Squeeze)
-   [Sub](#Sub)
-   [Sign](#Sign)
-   [Sin](#Sin)
-   [Sinh](#Sinh)
-   [Size](#Size)
-   [Sum](#Sum)
-   [Tanh](#Tanh)
-   [Tile](#Tile)
-   [ThresholdedRelu](#ThresholdedRelu)
-   [TopK](#TopK)
-   [Transpose](#Transpose)
-   [pad](#pad)
-   [Pow](#Pow)
-   [Unsqueeze](#Unsqueeze)
-   [Where](#Where)
## Abs<a name="Abs"></a>

### 功能<a name="section12725193815114"></a>

对输入张量取绝对值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double、int8、int16、int32、int64

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致"	

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Acos<a name="Acos"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的反余弦值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Acosh<a name="Acosh"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的反双曲余弦值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## AdaptiveMaxPool2D<a name="AdaptiveMaxPool2D"></a>

### 功能<a name="section12725193815114"></a>

对输入进行2d自适应最大池化计算

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、float64等

【属性】

一个属性：

output\_size:int型数组，指定输出的hw的shape大小

【输出】

两个输出

y：一个tensor，数据类型：与x类型一致

argmax:一个tensor，数据类型：int

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## Add<a name="Add"></a>

### 功能<a name="section12725193815114"></a>

按元素求和按元素求和

### 边界<a name="section9981612134"></a>

【输入】

两个输入

A：一个张量，数据类型：int8、int16、int32、int64、uint8、float32、float16、double

B：一个张量，数据类型与A相同

【输出】

C：一个张量，数据类型与A相同

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Addcmul<a name="Addcmul"></a>

### 功能<a name="section12725193815114"></a>

元素级计算\(x2 \* x3\) \* value + input\_data

### 边界<a name="section9981612134"></a>

【输入】

四个输入

input\_data：一个tensor，数据类型：float16、float32、int32、int8、uint8

x1: 一个tensor，类型与inpu\_data相同

x2: 一个tensor，类型与inpu\_data相同

value: 一个tensor，类型与inpu\_data相同

【输出】

一个输出

y：一个tensor，数据类型：y与输入相同

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## AffineGrid<a name="AffineGrid"></a>

### 功能<a name="section12725193815114"></a>

给定一批矩阵，生成采样网络

### 边界<a name="section9981612134"></a>

【输入】

俩个输入

theta：一个tensor，数据类型：float16、float32

output\_size：一个tensor，数据类型：int32

【属性】

一个属性：

align\_corners:bool型

【输出】

一个输出

y：一个tensor，数据类型：int

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## And<a name="And"></a>

### 功能<a name="section12725193815114"></a>

逻辑与

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：bool

x2：一个tensor，数据类型：bool

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Argmax<a name="Argmax"></a>

### 功能<a name="section12725193815114"></a>

返回指定轴上最大值所对应的索引

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Argmin<a name="Argmin"></a>

### 功能<a name="section12725193815114"></a>

返回输入张量指定轴上最小值对应的索引

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## AscendRequantS16<a name="AscendRequantS16"></a>

### 功能<a name="section12725193815114"></a>

重新量化算子

### 边界<a name="section9981612134"></a>

【输入】

两个必选输入，一个可选输入

x0：一个tensor，数据类型：int16

req\_scale：一个tensor，数据类型：uint64

x1：一个tensor，数据类型：int16

【属性】

两个属性：

dual\_output:bool型

relu\_flag:bool型

【输出】

两个输出

y0：一个tensor，数据类型：int8

y1：一个tensor，数据类型：int16

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## AscendRequant<a name="AscendRequant"></a>

### 功能<a name="section12725193815114"></a>

重新量化算子

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x0：一个tensor，数据类型：int32

req\_scale：一个tensor，数据类型：uint64

【属性】

一个属性：

relu\_flag:bool型

【输出】

一个输出

y：一个tensor，数据类型：int8

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## AscendQuant<a name="AscendQuant"></a>

### 功能<a name="section12725193815114"></a>

量化算子

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16，float32

【属性】

四个属性：

offset:float型

scale:float型

sqrt\_mode：bool型

round\_mode: string

【输出】

一个输出

y：一个tensor，数据类型：int8

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## AscendDequantS16<a name="AscendDequantS16"></a>

### 功能<a name="section12725193815114"></a>

反量化算子

### 边界<a name="section9981612134"></a>

【输入】

两个必选输入，一个可选输入

x0：一个tensor，数据类型：int32

req\_scale：一个tensor，数据类型：uint64

x1：一个tensor，数据类型：int16

【属性】

一个属性：

relu\_flag:bool型

【输出】

一个输出

y：一个tensor，数据类型：int16

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## AscendDequant<a name="AscendDequant"></a>

### 功能<a name="section12725193815114"></a>

反量化算子

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x0：一个tensor，数据类型：int32

deq\_scale：一个tensor，数据类型：uint64,float16

【属性】

三个属性：

sqrt\_mode：bool型

relu\_flag:bool型

dtype：float

【输出】

一个输出

y：一个tensor，数据类型：float16，float

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## AscendAntiQuant<a name="AscendAntiQuant"></a>

### 功能<a name="section12725193815114"></a>

反量化算子

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：int8

【属性】

四个属性：

offset:float型

scale:float型

sqrt\_mode：bool型

round\_mode: string

【输出】

一个输出

y：一个tensor，数据类型：float16，float

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## Asin<a name="Asin"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的反正弦

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x1：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Asinh<a name="Asinh"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量双曲反正弦

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

y：一个tenso，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Atan<a name="Atan"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的反正切值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Atanh<a name="Atanh"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的双曲反正切

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## AveragePool<a name="AveragePool"></a>

### 功能<a name="section12725193815114"></a>

平均池化层

### 边界<a name="section9981612134"></a>

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

strides\[0\]或者strides\[1\]取值步长大于63时，会走aicpu芯片，性能会下降；

kernel\_shape\_H或kernel\_shape\_W取值超过\[1,255\]，或者ksizeH \* ksizeW \> 256时，也会走aicpu，导致性能下降；

1 <= input\_w <= 4096；

当输入张量的N是一个质数时，N应当小于65535；

ceil\_mode参数仅在auto\_pad='NOTSET'时生效；

不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时fp32类型输入；

auto\_pad属性值SAME\_UPPER, SAME\_LOWER统一使用的TBE的SAME属性，即TBE算子没有根据这个属性区分pad的填充位置，可能会带来精度问题

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## BatchNormalization<a name="BatchNormalization"></a>

### 功能<a name="section12725193815114"></a>

标准化张量

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## BatchMatMul<a name="BatchMatMul"></a>

### 功能<a name="section12725193815114"></a>

将两个输入执行矩阵乘

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16，float，int32

x2：一个tensor，数据类型：float16，float，int32

【属性】

两个属性：

adj\_x1：bool型

adj\_x2:bool型

【输出】

一个输出

y：一个tensor，数据类型：float16，float，int32

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## BatchMultiClassNMS<a name="BatchMultiClassNMS"></a>

### 功能<a name="section12725193815114"></a>

为输入boxes和输入score计算nms

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## Cast<a name="Cast"></a>

### 功能<a name="section12725193815114"></a>

将输入数据的type转换为指定的type

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor

【输出】

y：一个tensor，输出的数据类型为属性指定的类型，数据类型：bool、float16、float32、int8、int32、uint8等

【属性】

to：数据类型：int，必选，指定目标数据类型，取值范围：在指定的数据类型范围内

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Ceil<a name="Ceil"></a>

### 功能<a name="section12725193815114"></a>

对输入张量向上取整

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Celu<a name="Celu"></a>

### 功能<a name="section12725193815114"></a>

连续可微的指数线性单位:对输入张量X按元素执行线性单位，使用公式:

max\(0,x\) + min\(0,alpha\*\(exp\(x/alpha\)-1\)\)

### 边界<a name="section9981612134"></a>

【输入】

X：tensor\(float\)

【输出】

Y：tensor\(float\)

【属性】

alpha：float，默认值：1.0

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Concat<a name="Concat"></a>

### 功能<a name="section12725193815114"></a>

对多个张量Concat

### 边界<a name="section9981612134"></a>

【输入】

inputs：多个输入张量，数据类型：float16、float32、int32、uint8、int16、int8、int64、qint8、quint8、qint32、uint16、uint32、uint64、qint16、quint16

【输出】

concat\_result：张量，与输入张量类型一致

【属性】

axis：指定哪一个轴进行concat操作，负数表示从后往前对维度计数，取值范围为\[-r, r - 1\]，r=rank\(inputs\)

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Clip<a name="Clip"></a>

### 功能<a name="section12725193815114"></a>

将张量值剪辑到指定的最小值和最大值之间

### 边界<a name="section9981612134"></a>

【输入】

三个输入

X ：一个张量，数据类型：float16、float32、int32

min：一个scalar

max：一个scalar

【输出】

一个输出

Y：一个张量，剪辑后的输出，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ConvTranspose<a name="ConvTranspose"></a>

### 功能<a name="section12725193815114"></a>

转置卷积

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Cumsum<a name="Cumsum"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量在给定axis上面的累加和

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Conv<a name="Conv"></a>

### 功能<a name="section12725193815114"></a>

卷积

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

## Constant<a name="Constant"></a>

### 功能<a name="section12725193815114"></a>

构建constant节点张量

### 边界<a name="section9981612134"></a>

【输入】

无

【输出】

一个输出

Y：输出张量，和提供的tensor值一致

【属性】

value：输出张量的值

【约束】

sparse\_value：不支持

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ConstantOfShape<a name="ConstantOfShape"></a>

### 功能<a name="section12725193815114"></a>

用给定的值和shape生成张量

### 边界<a name="section9981612134"></a>

【输入】

x：1D的int64的tensor，表示输出数据的shape，所有的值必须大于0

【输出】

y：一个tensor，shape由输入指定，如果属性value指定了值，那输出的值和数据类型就等于value指定的值，如果属性value不指定，输出tensor的值默认为0，数据类型默认为float32

【属性】

value：指定输出tensor的数据和类型

【约束】

x：1<=len\(shape\)<=8

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

## Cos<a name="Cos"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的余弦值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Cosh<a name="Cosh"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的双曲余弦

### 边界<a name="section9981612134"></a>

【输入】

一个输入

X1：一个tensor，数据类型：float16、float、double

【输出】

一个输出

y：一个张量，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Det<a name="Det"></a>

### 功能<a name="section12725193815114"></a>

计算方形矩阵行列式

### 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## DepthToSpace<a name="DepthToSpace"></a>

### 功能<a name="section12725193815114"></a>

将数据由深度重排到空间数据块

### 边界<a name="section9981612134"></a>

【输入】

1个输入

input:format为NCHW的tensor输入，类型：float16、float32,double，int32,int64等

【输出】

1个输出

output：一个张量,shape为\[N, C/\(blocksize \* blocksize\), H \* blocksize, W \* blocksize\]

【属性】

blocksize：int，必选 指定被移动的块的大小

mode: string 指定是depth-column-row还是column-row-depth排列，默认DCR

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Div<a name="Div"></a>

### 功能<a name="section12725193815114"></a>

按元素进行除法运算

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16、float32、double、int32、int64

x2：一个tensor，数据类型：float16、float32、double、int32、int64

【输出】

一个输出

y：一个tensor，数据类型和输入一致

【约束】

输入、输出的type相同

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Dropout<a name="Dropout"></a>

### 功能<a name="section12725193815114"></a>

拷贝或者屏蔽输入数据

### 边界<a name="section9981612134"></a>

【输入】

1-3个输入

data:tensor输入，类型：float16、float32,double等

ratio:可选输入，类型：float16、float32,double等

training\_mode:可选输入，类型：布尔型

【输出】

1-2个输出

output：一个张量

mask: 一个张量

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## elu<a name="elu"></a>

### 功能<a name="section12725193815114"></a>

elu激活函数

### 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【属性】

alpha：float，默认为1.0，含义：系数

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## EmbeddingBag<a name="EmbeddingBag"></a>

### 功能<a name="section12725193815114"></a>

计算embedding函数的反向输出

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## Equal<a name="Equal"></a>

### 功能<a name="section12725193815114"></a>

判断两个输入张量对应位置是否相等

### 边界<a name="section9981612134"></a>

【输入】

两个输入

X1：一个tensor

X2：一个tensor

【输出】

一个输出

y：一个tensor ，数据类型：bool

【约束】

输入X1、X2的数据类型和格式相同，支持如下数据类型：bool、uint8、int8、int16、int32、int64、float16、float32、double

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Erf<a name="Erf"></a>

### 功能<a name="section12725193815114"></a>

高斯误差函数

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型和格式与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

## Exp<a name="Exp"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的指数

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Expand<a name="Expand"></a>

### 功能<a name="section12725193815114"></a>

将输入tensor广播到指定shape

### 边界<a name="section9981612134"></a>

【输入】

2个输入

input：tensor，数据类型：float16、float32

shape：tensor，数据类型：int64

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【约束】

需要修改模型将输入shape由placeholder改为const类型，可以使用onnxsimplifier简化模型

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Flatten<a name="Flatten"></a>

### 功能<a name="section12725193815114"></a>

将张量展平

### 边界<a name="section9981612134"></a>

【输入】

input：多维张量，数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32

【输出】

具有输入张量的内容的2D张量

【属性】

axis：int，该参数暂不支持负值索引

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Floor<a name="Floor"></a>

### 功能<a name="section12725193815114"></a>

对输入张量向下取整

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Gather<a name="Gather"></a>

### 功能<a name="section12725193815114"></a>

根据相应的轴从“x”中收集切片

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## GatherND<a name="GatherND"></a>

### 功能<a name="section12725193815114"></a>

将输入数据切片输出

### 边界<a name="section9981612134"></a>

【输入】

2个输入

data:秩r\>=1的tensor输入，类型：float16, float32, double, int32, int64等

indices：int64的索引张量,秩q\>=1

【输出】

1个输出

output：一个张量, 秩为q + r - indices\_shape\[-1\] - 1

【属性】

batch\_dims：int，默认为0 批处理轴的数量

【约束】

不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时double的输入

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v11/v12/v13

## GatherElements<a name="GatherElements"></a>

### 功能<a name="section12725193815114"></a>

获取索引位置的元素产生输出

### 边界<a name="section9981612134"></a>

【输入】

2个输入

input:秩大于1的tensor输入，类型：float16、float32,double，int32,int64等

indices：int32/int64的索引张量

【输出】

1个输出

output：一个张量,与indices的shape相同

【属性】

axis：int，默认为0 指定聚集的轴

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Gemm<a name="Gemm"></a>

### 功能<a name="section12725193815114"></a>

全连接层

### 边界<a name="section9981612134"></a>

【输入】

A：2D矩阵张量，数据类型：float16、float32

B：2D矩阵张量，数据类型：float16、float32

C：偏差，可选，该参数暂不支持

【输出】

Y：2D矩阵张量，数据类型：float16、float32

【属性】

transA：布尔型，是否A需要转置

transB：布尔型，是否B需要转置

alpha：float，该参数暂不支持

beta：float，该参数暂不支持

【约束】

v8/v9/v10版本不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时fp32类型输入

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## GlobalAveragePool<a name="GlobalAveragePool"></a>

### 功能<a name="section12725193815114"></a>

全局平均池化

### 边界<a name="section9981612134"></a>

【输入】

X：一个张量，数据类型：float16、float32，格式为NCHW

【输出】

Y：池化输出张量，数据类型与X相同，格式为NCHW

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## GlobalMaxPool<a name="GlobalMaxPool"></a>

### 功能<a name="section12725193815114"></a>

全局最大池化算子

### 边界<a name="section9981612134"></a>

【输入】

1个输入

x:前一个节点的输出tensor，类型：float16, float32, double

【输出】

1个输出

output：池化后的张量

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Greater<a name="Greater"></a>

### 功能<a name="section12725193815114"></a>

按元素比较输入x1和x2的大小，若x1\>x2，对应位置返回true

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16、float32、int32、int8、uint8

x2：一个tensor，数据类型：float16、float32、int32、int8、uint8

【输出】

一个输出

y：一个tensor，数据类型：bool

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## GreaterOrEqual<a name="GreaterOrEqual"></a>

### 功能<a name="section12725193815114"></a>

按元素比较输入x1和x2的大小，若x1\>=x2，对应位置返回true

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16、float32、int32、int8、uint8等

x2：一个tensor，数据类型：float16、float32、int32、int8、uint8等

【输出】

一个输出

y：一个tensor，数据类型：bool

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v12

## Gru<a name="Gru"></a>

### 功能<a name="section12725193815114"></a>

计算单层GRU

### 边界<a name="section9981612134"></a>

【输入】

3-6个输入

X: 类型：float16, float32, double, int32, int64等

W:

R:

B:

sequence\_lens:

initial\_h:

【输出】

0-2个输出

Y:

Y\_h:

【属性】

activation\_alpha:

activation\_beta:

activations:

clip:

direction:

hidden\_size:

layout:

linear\_before\_reset:

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## HardSigmoid<a name="HardSigmoid"></a>

### 功能<a name="section12725193815114"></a>

HardSigmoid接受一个输入数据\(张量\)并生成一个输出数据\(张量\)，HardSigmoid函数y = max\(0, min\(1, alpha \* x + beta\)\)应用于张量元素方面。

### 边界<a name="section9981612134"></a>

【输入】

1个输入

X：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

【输出】

1个输出

Y：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

【属性】

alpha：float，默认值：0.2

beta：float，默认值：0.2

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v1/v6/v8/v9/v10/v11/v12/v13

## hardmax<a name="hardmax"></a>

### 功能<a name="section12725193815114"></a>

计算hardmax结果，如果元素是指定axis的最大元素则设为1，否则为0

### 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，rank=2，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【属性】

axis：int，默认为-1，含义：指定计算轴

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Identity<a name="Identity"></a>

### 功能<a name="section12725193815114"></a>

恒等操作

### 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## If<a name="If"></a>

### 功能<a name="section12725193815114"></a>

逻辑控制判断算子

### 边界<a name="section9981612134"></a>

【输入】

一个输入

cond：If op的条件

两个属性

else\_branch:条件为假的分支

then\_branch：条件为真的分支

【输出】

一到多个输出

y：tensor或者tensor序列

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Less<a name="Less"></a>

### 功能<a name="section12725193815114"></a>

按元素比较输入x1和x2的大小，若x1<x2，对应位置返回true

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16、float32、int32、int8、uint8

x2：一个tensor，数据类型：float16、float32、int32、int8、uint8

【输出】

一个输出

y：一个tensor，数据类型：bool

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## LeakyRelu<a name="LeakyRelu"></a>

### 功能<a name="section12725193815114"></a>

对输入张量用leakrelu函数激活

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y： 一个tensor，数据类型和shape与输入一致

【属性】

alpha：数据类型为float，默认0.01，表示leakage系数

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## LessOrEqual<a name="LessOrEqual"></a>

### 功能<a name="section12725193815114"></a>

小于等于计算

### 边界<a name="section9981612134"></a>

【输入】

2个输入

x：tensor，数据类型：float16、float32

y：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的shape,数据类型:bool

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v12/v13

## Log<a name="Log"></a>

### 功能<a name="section12725193815114"></a>

计算输入的自然对数

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## LogSoftMax<a name="LogSoftMax"></a>

### 功能<a name="section12725193815114"></a>

对输入张量计算logsoftmax值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

【属性】

axis：数据类型为int；指定计算的轴，取值范围：\[-r, r-1\]，r为输入的秩

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## LpNormalization<a name="LpNormalization"></a>

### 功能<a name="section12725193815114"></a>

给定一个矩阵，沿给定的轴应用LpNormalization。

### 边界<a name="section9981612134"></a>

【输入】

1个输入

input：，类型：tensor\(float16\), tensor\(float\)

【输出】

1个输出

output：，类型：tensor\(float16\), tensor\(float\)

【属性】

axis：int，默认值：-1

p：int，默认值：2

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v1/v8/v9/v10/v11/v12/v13

## LRN<a name="LRN"></a>

### 功能<a name="section12725193815114"></a>

对输入张量做局部响应归一化

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## LSTM<a name="LSTM"></a>

### 功能<a name="section12725193815114"></a>

计算单层LSTM。这个操作符通常通过一些自定义实现\(如CuDNN\)来支持。

### 边界<a name="section9981612134"></a>

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

clip: float

direction: string，默认值：forward

hidden\_size: int

input\_forget: int，默认值：0

layout: int，默认值：0

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## MatMul<a name="MatMul"></a>

### 功能<a name="section12725193815114"></a>

矩阵乘

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个2D的tensor，数据类型：float16

x2：一个2D的tensor，数据类型：float16

【输出】

一个输出

y：一个2D的tensor，数据类型：float16

【约束】

仅支持1-6维输入

只支持ND和2D的矩阵乘

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Max<a name="Max"></a>

### 功能<a name="section12725193815114"></a>

元素级比较输入tensor的大小

### 边界<a name="section9981612134"></a>

【输入】

多个输入\(1-∞\)

data\_0:tensor的列表，类型：float16、float32,int8,int16,int32等

【输出】

一个输出

max：一个张量，和输入x同样的type和shape（广播后的shape）

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## MaxPool<a name="MaxPool"></a>

### 功能<a name="section12725193815114"></a>

最大池化

### 边界<a name="section9981612134"></a>

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

strides\[0\]或者strides\[1\]取值步长大于63时，会走aicpu芯片，性能会下降；

kernel\_shape\_H或kernel\_shape\_W取值超过\[1,255\]，或者ksizeH \* ksizeW \> 256时，也会走aicpu，导致性能下降；

1 <= input\_w <= 4096

当输入张量的N是一个质数时，N应小于65535

2D tensor输入不支持dilations

auto\_pad属性是VALID时，ceil\_mode属性值必须为0

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## MaxRoiPool<a name="MaxRoiPool"></a>

### 功能<a name="section12725193815114"></a>

ROI最大池消耗一个输入张量X和感兴趣区域\(ROI\)，以便在每个ROI上应用最大池，从而产生输出的4-D形状张量\(num\_roi, channels, pooled\_shape\[0\]， pooled\_shape\[1\]\)。

### 边界<a name="section9981612134"></a>

【输入】

X：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

rois：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

【输出】

Y：，类型：tensor\(float16\), tensor\(float\), tensor\(double\)

【属性】

pooled\_shape: list of ints

spatial\_scale: float，默认值：1.0

【约束】

不支持fp64输入

不支持atc工具参数--precision\_mode=must\_keep\_origin\_dtype时fp32类型输入

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/13

## Mean<a name="Mean"></a>

### 功能<a name="section12725193815114"></a>

每个输入张量的元素均值\(支持numpy风格的广播\)。所有输入和输出必须具有相同的数据类型。该操作符支持多向\(即numpy风格\)广播。

### 边界<a name="section9981612134"></a>

【输入1-∞】

data\_0：，类型：tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

mean：，类型：tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## MeanVarianceNormalization<a name="MeanVarianceNormalization"></a>

### 功能<a name="section12725193815114"></a>

一个均值标准化函数:使用公式对输入张量X进行均值方差标准化：\(X-EX\)/sqrt\(E\(X-EX\)^2\)

### 边界<a name="section9981612134"></a>

【输入】

X：，类型：tensor\(float16\), tensor\(float\), tensor\(bfloat16\)

【输出】

Y：，类型：tensor\(float16\), tensor\(float\), tensor\(bfloat16\)

【属性】

axes: list of ints，默认值：\['0', '2', '3'\]

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

## Min<a name="Min"></a>

### 功能<a name="section12725193815114"></a>

计算输入tensors的最小值

### 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor列表，数据类型：float16、float32

【输出】

一个输出

y：计算出最小值的tensor

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Mod<a name="Mod"></a>

### 功能<a name="section12725193815114"></a>

执行元素二进制模数\(支持numpy风格的广播\)。余数的符号与除数的符号相同。

### 边界<a name="section9981612134"></a>

【输入】

A：，类型：tensor\(uint8\), tensor\(uint16\), tensor\(uint32\), tensor\(uint64\), tensor\(int8\), tensor\(int16\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

B：，类型：tensor\(uint8\), tensor\(uint16\), tensor\(uint32\), tensor\(uint64\), tensor\(int8\), tensor\(int16\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

C：，类型：tensor\(uint8\), tensor\(uint16\), tensor\(uint32\), tensor\(uint64\), tensor\(int8\), tensor\(int16\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【属性】

fmod：，类型：int，默认值：0

【约束】

当输入类型为浮点时，fmod不支持为0

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12/v13

## Mul<a name="Mul"></a>

### 功能<a name="section12725193815114"></a>

矩阵点乘

### 边界<a name="section9981612134"></a>

【输入】

A：一个张量，数据类型：float16、float32、uint8、int8、int16、int32

B：一个张量，数据类型：float16、float32、uint8、int8、int16、int32

【输出】

C：一个张量，数据类型与输入张量一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Multinomial<a name="Multinomial"></a>

### 功能<a name="section12725193815114"></a>

返回Multinomial采样结果矩阵

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Neg<a name="Neg"></a>

### 功能<a name="section12725193815114"></a>

求输入的负数

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、int32

【输出】

一个输出

y：一个tensor，数据类型与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## NonMaxSuppression<a name="NonMaxSuppression"></a>

### 功能<a name="section12725193815114"></a>

过滤掉与先前选定的框有较高重叠的“交集-并集”\(IOU\)框。移除得分小于score\_threshold的边界框。边界框格式由属性center\_point\_box表示。注意，该算法不知道原点在坐标系中的位置，更普遍地说，它对坐标系的正交变换和平移是不变的;因此，平移或反射坐标系统的结果在相同的方框被算法选择。selected\_indices输出是一组整数，索引到表示所选框的边界框的输入集合中。然后，可以使用Gather或gatherd操作获得与所选索引对应的边框坐标。

### 边界<a name="section9981612134"></a>

【输入2-5】

boxes: tensor\(float\)

scores: tensor\(float\)

max\_output\_boxes\_per\_class: tensor\(int64\)

iou\_threshold: tensor\(float\)

score\_threshold: tensor\(float\)

【输出】

selected\_indices: tensor\(int64\)

【属性】

center\_point\_box: int 默认值：0

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12/v13

## NonZero<a name="NonZero"></a>

### 功能<a name="section12725193815114"></a>

返回非零元素的索引（按行大顺序-按维）

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、int32、int8、uint8等

【输出】

一个输出

y：一个tensor，数据类型：int64

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

## Not<a name="Not"></a>

### 功能<a name="section12725193815114"></a>

逻辑非

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：bool

【输出】

一个输出

y：一个tensor，数据类型：bool

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## OneHot<a name="OneHot"></a>

### 功能<a name="section12725193815114"></a>

根据输入生成一热编码张量

### 边界<a name="section9981612134"></a>

【输入】

三个输入

indices：一个tensor，数据类型：float16、float32、int32、int8等

depth：一个tensor，数据类型：float16、float32、int32、int8等

valus：一个tensor，数据类型：float16、float32、int32、int8等

【属性】

一个属性

axis:（可选）添加一热表示的轴

【输出】

一个输出

y：一个tensor，数据类型与value输入的类型一致

【约束】

算子属性不支持axis<-1

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v9/v10/v11/v12/v13

## Or<a name="Or"></a>

### 功能<a name="section12725193815114"></a>

逻辑或

### 边界<a name="section9981612134"></a>

【输入】

两个输入

X1：一个tensor，数据类型：bool

X2：一个tensor，数据类型：bool

【输出】

一个输出

y：一个tensor，数据类型：bool

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## randomUniform<a name="randomUniform"></a>

### 功能<a name="section12725193815114"></a>

生成具有从均匀分布绘制的随机值的张量

### 边界<a name="section9981612134"></a>

【属性】

五个属性

dtype:int类型，指明输出类型

high：float型，指明上边界

low:float型，指明下边界

seed：\(可选\)，随机种子

shape：输出的形状

【输出】

一个输出

y：一个tensor，数据类型与dtype属性指定类型一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Range<a name="Range"></a>

### 功能<a name="section12725193815114"></a>

产生一个连续序列的tensor

### 边界<a name="section9981612134"></a>

【输入】

3个输入

start：scalar，数据类型：float16、float32

limit：scalar，数据类型：float16、float32

delta：scalar，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Reciprocal<a name="Reciprocal"></a>

### 功能<a name="section12725193815114"></a>

将输入张量取倒数

### 边界<a name="section9981612134"></a>

【输入】

一个输入

lx：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ReduceL1<a name="ReduceL1"></a>

### 功能<a name="section12725193815114"></a>

沿所提供的轴计算输入张量元素的L1范数。如果keepdim等于1，得到的张量的秩与输入的相同。如果keepdim等于0，那么得到的张量就会被精简维数。上述行为与numpy类似，只是numpy默认keepdim为False而不是True。

### 边界<a name="section9981612134"></a>

【输入】

data：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

reduced：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【属性】

axes: list of ints

keepdims: int，默认值：1

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ReduceL2<a name="ReduceL2"></a>

### 功能<a name="section12725193815114"></a>

沿所提供的轴计算输入张量元素的L2范数。如果keepdim等于1，得到的张量的秩与输入的相同。如果keepdim等于0，那么得到的张量就会被精简维数。上述行为与numpy类似，只是numpy默认keepdim为False而不是True。

### 边界<a name="section9981612134"></a>

【输入】

data：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

reduced：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【属性】

axes: list of ints

keepdims: int，默认值：1

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ReduceMin<a name="ReduceMin"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量指定方向的最小值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型：float16、float32

【属性】

axes：数据类型为listInt；含义：指定计算轴；取值范围：\[-r, r-1\]，r是输入数据的秩

keepdims：数据类型为int；含义：是否保留缩减后的维度；默认为1

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ReduceMean<a name="ReduceMean"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的指定维度的元素的均值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和format

【属性】

axes：一个1D的整数列表，含义：指定精减的维度，取值范围为\[-r, r - 1\]，r是输入矩阵的秩

keepdims：数据类型为int，默认为1，含义：是否保留缩减后的维度

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ReduceProd<a name="ReduceProd"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的元素沿所提供的轴的乘积。如果keepdim等于1，得到的张量的秩与输入的相同。如果keepdim等于0，那么得到的张量就会被精简维数。

### 边界<a name="section9981612134"></a>

【输入】

data：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

reduced：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【属性】

axes: list of ints

keepdims: int，默认值：1

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ReduceSumSquare<a name="ReduceSumSquare"></a>

### 功能<a name="section12725193815114"></a>

沿所提供的轴计算输入张量元素的平方和。如果keepdim等于1，得到的张量的秩与输入的相同。如果keepdim等于0，那么得到的张量就会被精简维数。上述行为与numpy类似，只是numpy默认keepdim为False而不是True。

### 边界<a name="section9981612134"></a>

【输入】

data：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【输出】

reduced：tensor\(uint32\), tensor\(uint64\), tensor\(int32\), tensor\(int64\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(bfloat16\)

【属性】

axes: list of ints

keepdims: int，默认值：1

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v1/v8/v9/v10/v11/v12/v13

## Resize<a name="Resize"></a>

### 功能<a name="section12725193815114"></a>

调整输入tensor大小

### 边界<a name="section9981612134"></a>

【输入】

4个输入

x：一个tensor，数据类型：float16、float32

roi: 被输入图像归一化的1Dtensor，\[start1, ..., startN, end1, ..., endN\]，数据类型：float16、float32

scales:与输入x的秩相等的数组

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12

## Relu<a name="Relu"></a>

### 功能<a name="section12725193815114"></a>

整流线性单位函数

### 边界<a name="section9981612134"></a>

【输入】

X：输入张量，数据类型：float32、int32、uint8、int16、int8、uint16、float16、qint8

【输出】

Y：输出张量，数据类型与X一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ReduceSum<a name="ReduceSum"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量指定维度的元素的和

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x的type和format相同

【属性】

axes：一个1D的整数列表，含义：指定精减的维度，取值范围为\[-r, r - 1\]（r是输入矩阵的秩）

keepdims：数据类型为int，默认为1，含义：是否保留缩减后的维度

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ReduceMax<a name="ReduceMax"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量指定方向的最大值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、int32

【输出】

一个输出

y：一个tensor，数据类型：float16、float32、int32

【属性】

axes：数据类型为listInt；含义：指定计算轴；取值范围：\[-r, r-1\]，r是输入数据的秩

keepdims：数据类型为int；含义：是否保留缩减后的维度；默认为1

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Reshape<a name="Reshape"></a>

### 功能<a name="section12725193815114"></a>

改变输入维度

### 边界<a name="section9981612134"></a>

【输入】

两个输入

data：一个张量

shape：一个张量，定义了输出张量的形状，int64

【输出】

reshaped：一个张量

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ReverseSequence<a name="ReverseSequence"></a>

### 功能<a name="section12725193815114"></a>

根据指定长度对batch序列进行排序

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12/v13

## RoiExtractor<a name="RoiExtractor"></a>

### 功能<a name="section12725193815114"></a>

从特征映射列表中获取ROI特征矩阵

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

自定义算子，无对应onnx版本

## RoiAlign<a name="RoiAlign"></a>

### 功能<a name="section12725193815114"></a>

在每个roi区域进行池化处理

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12/v13

## Round<a name="Round"></a>

### 功能<a name="section12725193815114"></a>

对输入张量做四舍五入的运算

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## PRelu<a name="PRelu"></a>

### 功能<a name="section12725193815114"></a>

PRelu激活函数

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x：一个tensor，数据类型：float16、float32

slope：slope张量，数据类型和输入x一致

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【约束】

slope必须是1维，当输入x的shape是1维时，slope的维度值必须为1；输入x的shape是其他维度时，slope的维度值可以为1或者为输入x的shape\[1\]

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ScatterND<a name="ScatterND"></a>

### 功能<a name="section12725193815114"></a>

创建data的拷贝，同时在指定indices处根据updates更新

### 边界<a name="section9981612134"></a>

【输入】

3个输入

data：tensor，rank \>= 1，数据类型：float16、float32

indices：tensor，rank \>= 1，数据类型：int64

updates：tensor，rank = q + r - indices\_shape\[-1\] - 1，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v11

## Selu<a name="Selu"></a>

### 功能<a name="section12725193815114"></a>

在元素级别使用指数线性单位函数y = gamma \* \(alpha \* e^x - alpha\) for x <= 0, y = gamma \* x for x \> 0 生成张量

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x:fp16,fp32,double类型的tensor

两个属性

alpha:乘数因子

gamma：乘数因子

【输出】

一个输出

y：与输入类型相同的tensor

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Shape<a name="Shape"></a>

### 功能<a name="section12725193815114"></a>

获取输入tensor的shape

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor

【输出】

y：输入tensor的shape，数据类型为int64的tensor

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Sigmoid<a name="Sigmoid"></a>

### 功能<a name="section12725193815114"></a>

对输入做sigmoid

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：数据类型支持float16、float32

【输出】

一个输出

y：数据类型和输入x一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Slice<a name="Slice"></a>

### 功能<a name="section12725193815114"></a>

获取输入tensor的切片

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Softmax<a name="Softmax"></a>

### 功能<a name="section12725193815114"></a>

对输入进行softmax

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，类型和shape与输入x一致

【属性】

axis：Int，可选，表示进行softmax的方向，默认值为-1，范围为\[ -len\(x.shape\), len\(x.shape\)-1\]

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Softsign<a name="Softsign"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的softsign\(x/\(1+|x|\)\)

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Softplus<a name="Softplus"></a>

### 功能<a name="section12725193815114"></a>

计算softplus

### 边界<a name="section9981612134"></a>

【输入】

一个输入

X：1D的输入张量

【输出】

一个输出

Y：1D的张量

【约束】

数据类型仅支持float16、float32

输入、输出的数据类型一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## SpaceToDepth<a name="SpaceToDepth"></a>

### 功能<a name="section12725193815114"></a>

SpaceToDepth将空间数据块重新排列成深度。更具体地说，这个op输出一个输入张量的副本，其中高度和宽度维度的值移动到深度维度。

### 边界<a name="section9981612134"></a>

【输入】

input：tensor\(uint8\), tensor\(uint16\), tensor\(uint32\), tensor\(uint64\), tensor\(int8\), tensor\(int16\), tensor\(int32\), tensor\(int64\), tensor\(bfloat16\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(string\), tensor\(bool\), tensor\(complex64\), tensor\(complex128\)

【输出】

output：tensor\(uint8\), tensor\(uint16\), tensor\(uint32\), tensor\(uint64\), tensor\(int8\), tensor\(int16\), tensor\(int32\), tensor\(int64\), tensor\(bfloat16\), tensor\(float16\), tensor\(float\), tensor\(double\), tensor\(string\), tensor\(bool\), tensor\(complex64\), tensor\(complex128\)

【属性】

blocksize: int

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Split<a name="Split"></a>

### 功能<a name="section12725193815114"></a>

将输入切分成多个输出

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Sqrt<a name="Sqrt"></a>

### 功能<a name="section12725193815114"></a>

计算元素的平方根

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor

【输出】

一个输出

y：一个tensor

【约束】

输入、输出的数据类型相同，支持的数据类型：float16、float32

如果x小于0，返回Nan

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Squeeze<a name="Squeeze"></a>

### 功能<a name="section12725193815114"></a>

从输入中去除尺寸为1的维度

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个张量，数据类型：float16、float32、double、uint8、uint16、uint32、uint64、int8、int16、int32、int64、bool

【输出】

y：一个tensor，数据类型和输入一致

【属性】

axes：一个数据类型为int32或者int64的整形列表，指定维度的维度值需要为1；取值范围为\[-r, r-1\]（r表示输入张量的秩，负数表示从后面计算维度）；含义：指定要去除的维度

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Sub<a name="Sub"></a>

### 功能<a name="section12725193815114"></a>

进行张量的减法运算

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor

x2：一个tensor

【输出】

一个输出

y：一个张量，数据类型和输入一致

【约束】

输入、输出的shape和dtype相同，支持的数据类型：int32、float16、float32

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Sign<a name="Sign"></a>

### 功能<a name="section12725193815114"></a>

逐元素计算输入tensor的符号

### 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Sin<a name="Sin"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量的正弦值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Sinh<a name="Sinh"></a>

### 功能<a name="section12725193815114"></a>

计算输入张量双曲正弦值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32、double

【输出】

一个输出

y：一个tensor，数据类型和shape与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Size<a name="Size"></a>

### 功能<a name="section12725193815114"></a>

计算输入tensor的元素个数

### 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个int64的scalar

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Sum<a name="Sum"></a>

### 功能<a name="section12725193815114"></a>

求和

### 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor序列，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Tanh<a name="Tanh"></a>

### 功能<a name="section12725193815114"></a>

计算输入的双曲正切值

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：float16、float32

【输出】

一个输出

y：一个tensor，数据类型与输入一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Tile<a name="Tile"></a>

### 功能<a name="section12725193815114"></a>

将输入张量沿指定维度重复

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x：一个tensor

repeats：一个1D的int64的tensor，size和输入的维度数一样

【输出】

一个输出

y：输出的tensor，type和维度与输入一致，output\_dim\[i\] = input\_dim\[i\] \* repeats\[i\]

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## ThresholdedRelu<a name="ThresholdedRelu"></a>

### 功能<a name="section12725193815114"></a>

当x \> alpha时y = x，否则y=0

### 边界<a name="section9981612134"></a>

【输入】

1个输入

x：tensor，数据类型：float16、float32

【输出】

一个输出

y：一个张量，和输入x同样的type和shape

【属性】

alpha：float，默认为1.0，含义：阈值

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v10/v11/v12/v13

## TopK<a name="TopK"></a>

### 功能<a name="section12725193815114"></a>

返回指定轴的k个最大或最小值

### 边界<a name="section9981612134"></a>

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

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Transpose<a name="Transpose"></a>

### 功能<a name="section12725193815114"></a>

转置

### 边界<a name="section9981612134"></a>

【输入】

data：一个张量，数据类型：float16、float32、int8、int16、int32、int64、uint8、uint16、uint32、uint64

【输出】

transposed：转置之后的张量

【属性】

perm：整数列表， 张量data的维度排列

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## pad<a name="pad"></a>

### 功能<a name="section12725193815114"></a>

对输入tensor做填充

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x：数据类型支持float16、float32、int32

pads：数据类型支持int32 、int64

【输出】

一个输出

y：数据类型和输入x一致

【约束】

当mode值为constant时，目前仅支持constant\_value=0

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Pow<a name="Pow"></a>

### 功能<a name="section12725193815114"></a>

计算输入x1的x2次幂

### 边界<a name="section9981612134"></a>

【输入】

两个输入

x1：一个tensor，数据类型：float16、float32、double、int32、int8、uint8

x2：一个tensor，数据类型和输入x1一致

【输出】

一个输出

y：数据类型和输入x1一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

## Unsqueeze<a name="Unsqueeze"></a>

### 功能<a name="section12725193815114"></a>

在输入张量（数据）的形状中插入一维项

### 边界<a name="section9981612134"></a>

【输入】

一个输入

x：一个tensor，数据类型：uint8、uint16、uint32、int8、int16、int32、float16、float32

【输出】

一个输出

y：一个tensor，数据类型和输入x一致

【属性】

axes：ListInt，表示在指定的维度进行插1维项，取值范围为\[-input\_rank, input\_rank\]，input\_rank为输入张量的秩，axes的内容不可以重复

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/10/v11/v12

## Where<a name="Where"></a>

### 功能<a name="section12725193815114"></a>

根据条件从两个输入中选择元素

### 边界<a name="section9981612134"></a>

【输入】

三个输入

condition，条件，数据类型：bool

x：一个tensor，条件为true时从x中选取元素，数据类型支持float16、float32、int8、int32、uint8

y：一个tensor，条件为false时从y中选取元素，和x的数据类型一致

【输出】

一个tensor，数据类型和输入x一致

### 支持的ONNX版本<a name="section13311501226"></a>

Opset v8/v9/v10/v11/v12/v13

