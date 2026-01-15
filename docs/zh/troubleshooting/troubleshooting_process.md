# 故障处理流程

本文主要以开发者在执行推理、训练过程中可能遇到的各类异常故障现象为入口，提供自助式问题定位、问题处理方法，方便开发者快速定位并解决故障，内容包括：**屏幕打印的错误码信息及处理方法**、**一键式日志收集**以及**各类问题定位工具使用**。

**故障处理总体流程主要包括以下过程：收集故障信息、分析故障原因、故障排除**。具体实施过程如[图1](#故障处理流程图)所示。

**图 1**  故障处理流程<a id="故障处理流程图"></a>  
![](figures/troubleshooting_process.png "故障处理流程")

-   参考“错误码”处理

    关于CANN软件“错误码”的详细介绍，请参见《CANN 故障处理》的“[错误码参考](https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/troubleshooting/troubleshooting_0225.html)”章节。

    关于torch\_npu插件“错误码”的详细介绍，请参见[Error Code介绍](error_codes_introduction.md)。

-   收集故障信息

    故障信息是故障处理的重要依据，故障处理人员应尽可能多地收集故障信息，包括但不限于日志、环境信息等。

    关于日志信息，一般采用自上而下的日志分析方法，根据业务流程逐步缩小到底层故障现象。

    关于日志级别的详细介绍，请参见《CANN 日志参考》中的“[设置日志级别](https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/logreference/logreference_0008.html)”章节。

    关于日志路径以及日志文件的详细介绍，请参见《CANN 日志参考》中的“[查看日志（Ascend EP）](https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/logreference/logreference_0002.html)”章节。

    关于回显信息，Ascend Extension for PyTorch的告警信息默认正常打印，集群场景下告警信息会正常打印在首节点的屏幕上。

    通过msnpureport工具将Device侧的系统日志传输到Host侧进行查看，具体请参见《[msnpureport工具使用](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-800t-a2-pid-254184887?category=reference-guides)》。

-   分析故障原因

    分析故障原因是指从众多可能原因中找出故障原因的过程。通过一定的方法或手段分析、比较各种可能的故障成因，不断排除可能因素，最终确定故障发生的具体原因。

-   故障排除

    故障排除是指根据不同的故障原因清除故障的过程。

-   记录故障处理过程

    故障排除后应记录故障处理要点，给出针对此类故障的防范和改进措施，避免同类故障再次发生。

> [!NOTE]  
> 您也可以将故障处理案例分享到[华为开发者社区论坛](https://www.hiascend.com/forum/)，分享您的经验，供其他开发者参考，形成良性循环，丰富社区内容，实现共同受益。
> 本文提供的故障处理步骤中涉及的第三方工具（如eseye u、Netron），均为举例，非必须工具，请根据您自己实际情况参考使用或替换成其他类似工具。

