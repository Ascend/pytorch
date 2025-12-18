# 回显信息

回显信息通常比较多，许多组件都会在屏幕上进行错误信息的输出，可以大致分为如下几类：

-   Python调用栈和异常信息
-   torch\_npu错误码
-   CANN软件错误码
-   原生框架报错信息

## Python调用栈和异常信息

Python报错时，会将当时的堆栈打印在屏幕上，用户可通过搜索关键字“Traceback”查看Python应用程序的堆栈。如果存在多个堆栈信息时，优先查看第一个Traceback，如[图1](#查看Python应用程序的堆栈信息)所示。

**图 1**  查看Python应用程序的堆栈信息<a id="查看Python应用程序的堆栈信息"></a>  
![](figures/viewing_stack_information_python_applications.png)

如上回显信息示例中，用户可以看到最后的调用栈在torch\_npu的`set_autocast_enabled`接口上，一般情况下您可以到[昇腾社区](https://gitcode.com/Ascend/pytorch/issues)提交issue获取帮助。

若最后的调用栈在原生torch上，则可以顺着堆栈向上找与昇腾相关的堆栈，若整个堆栈没有昇腾相关的堆栈，则排查模型训练脚本本身是否有问题。

## torch\_npu错误码

在模型训练过程中，打印的错误码信息可能因场景、模式、故障原因的不同而有所区别。因此，需要结合具体的报错信息以及plog日志等共同定位，torch\_npu错误码相关详细介绍可参见[Error Code介绍](error_codes_introduction.md)。

错误码在回显中的表示形式为：

\[ERROR\] \[%s\] \(PID:\[%s\], Device:\[%s\], RankID:\[%s\]\) ERR\[%s\]\[%s\] \[%s\] \[%s\]

## CANN软件错误码

由于场景不同、用例不同、发生故障的原因不同，造成打印的错误码信息有区别，因此，示例中以\[%s\]变量形式替代实际的打印日志，\[%s\]替代的实际日志以屏幕打印为准。CANN软件错误码相关详细介绍可参见《CANN 故障处理》的“[错误码参考](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/troubleshooting/troubleshooting_0225.html)”章节。

例如，E10035错误码在手册中的表示形式为：

E10035: \[PID:  _xxxxxx_\]  _时间戳_  The \[--dynamic\_batch\_size\], \[--dynamic\_image\_size\], or \[--dynamic\_dims\] argument has  \[%s\] profiles, which is less than the minimum \[%s\].

如[图2](#CANN软件错误码回显示例)所示，既有Python调用栈错误，又有torch\_npu错误码（ERR00100）以及CANN软件错误码（EZ3002）。用户可以从如下示例中看到CANN软件错误码在最前端，因此主要关注其报错信息。从报错信息中可以看到存在不支持的算子，一般情况下您可以获取日志后单击[Link](https://www.hiascend.com/support)联系技术支持。

**图 2**  CANN软件错误码回显示例<a id="CANN软件错误码回显示例"></a>  
![](figures/CANN_software_error_code.png "CANN软件错误码回显示例")

## 原生框架报错信息

查看Python调用栈中是否涉及原生框架报错信息，类似如下回显信息所示（以PyTorch 2.1.0为例），可单击[Link](https://github.com/pytorch/pytorch/issues)获取更多技术支持。

```Python
import torch
t1 = torch.tensor([[1, 2], [3, 4]],dtype=torch.bfloat16)
t2 = torch.tensor([2, 3],dtype=torch.bfloat16)
torch.isin(t1, t2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: Unsupported input type encountered for isin(): BFloat16
```

